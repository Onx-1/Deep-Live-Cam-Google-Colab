import zmq
import cv2
import modules.globals
import numpy as np
import threading
import time
import io
import modules.processors.frame.core
from tqdm import tqdm
from modules.typing import Face, Frame
from typing import Any,List
from modules.core import update_status
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
import zlib
import subprocess
from cv2 import VideoCapture
import queue
NAME = 'DLC.REMOTE-PROCESSOR'
STREAM_WIDTH = 960
STREAM_HEIGHT = 540
STREAM_CHANNELS = 3
STREAM_FRAME_SIZE = STREAM_WIDTH * STREAM_HEIGHT * STREAM_CHANNELS

context = zmq.Context()
SOCKET_TIMEOUT_MS = 15000
RECEIVE_TIMEOUT_S = 20
SOURCE_UPLOAD_LOCK = threading.Lock()


def configure_socket(socket: zmq.Socket) -> zmq.Socket:
    """
    Apply blocking guards so an unreachable tunnel cannot freeze the GUI thread.

    @param {zmq.Socket} socket - freshly created ZeroMQ socket.
    @returns {zmq.Socket} the same socket with timeouts and no lingering data.
    """
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, SOCKET_TIMEOUT_MS)
    socket.setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)
    return socket


def push_socket(address: str) -> zmq.Socket:
    """
    Create the request socket used to push payloads to the Colab server.

    @param {str} address - tunnel address of the Colab push endpoint.
    @returns {zmq.Socket} connected REQ socket with timeouts applied.
    """
    sender_sock = configure_socket(context.socket(zmq.REQ))
    sender_sock.connect(address)
    return sender_sock


def pull_socket(address: str) -> zmq.Socket:
    """
    Create the reply socket used to pull processed payloads from the Colab server.

    @param {str} address - tunnel address of the Colab pull endpoint.
    @returns {zmq.Socket} connected REP socket with timeouts applied.
    """
    sender_sock = configure_socket(context.socket(zmq.REP))
    sender_sock.connect(address)
    return sender_sock

def pre_check() -> bool:
    if not modules.globals.push_addr and not modules.globals.pull_addr:
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def stream_frame(temp_frame: Frame,stream_out: subprocess.Popen[bytes],stream_in: subprocess.Popen[bytes]) -> Frame:
    temp_framex = swap_face_remote(temp_frame,stream_out,stream_in)

    return temp_framex

def process_frame(source_frame: Frame, temp_frame: Frame)-> Frame:
    temp_framex = swap_frame_face_remote(source_frame,temp_frame)

    return temp_framex
def send_data(sender: zmq.Socket, face_bytes: bytes, metadata: dict, address: str) -> bool:
    """
    Stream a payload to the Colab server with acknowledgements.

    @param {zmq.Socket} sender - connected push socket.
    @param {bytes} face_bytes - raw frame bytes to transfer.
    @param {dict} metadata - frame descriptor sent before the chunks.
    @param {str} address - tunnel address used for error reporting.
    @returns {bool} True when the server acknowledged the whole payload.
    """
    chunk_size = max(1, len(face_bytes) // 10)
    total_chunk = (len(face_bytes) + chunk_size - 1) // chunk_size
    new_metadata = {'total_chunk': total_chunk}
    metadata.update(new_metadata)
    try:
        sender.send_json(metadata)
        ack = sender.recv_string()
        with tqdm(total=total_chunk, desc="Sending chunks", unit="chunk") as pbar:
            for i in range(total_chunk):
                chunk = face_bytes[i * chunk_size:(i + 1) * chunk_size]
                sender.send(chunk)
                ack = sender.recv_string()
                pbar.set_postfix_str(f'Chunk {i + 1}/{total_chunk} ack: {ack}')
                pbar.update(1)
        sender.send(b"END")
        final_reply_message = sender.recv_string()
        print(f"Received final reply: {final_reply_message}")
        return True
    except zmq.Again:
        print(f"[REMOTE] Timeout on {address}. Re-run the Colab live cell before each Live start, then paste the new tunnel addresses.")
        return False
    except zmq.ZMQError as error:
        print(f"[REMOTE] Socket error on {address}: {error}")
        return False
    finally:
        sender.close()
#client.py
def send_source_frame(source_face: Frame) -> bool:
    """
    Send the source face image to the Colab server.

    Bisa dipanggil ulang saat Live berjalan agar Colab memakai wajah sumber yang baru.

    @param {Frame} source_face - source face image loaded from disk.
    @returns {bool} True when the server acknowledged the source payload.
    """
    if not SOURCE_UPLOAD_LOCK.acquire(blocking=False):
        print('[REMOTE] upload source sebelumnya belum selesai, permintaan ini dilewati.')
        return False
    try:
        sender = push_socket(modules.globals.push_addr)
        source_face_bytes = source_face.tobytes()
        metadata = {
                'manyface':(modules.globals.many_faces),
                'dtype_source':str(source_face.dtype),
                'shape_source':source_face.shape,
                'size':'640x480',
                'fps':'60'
            }
        return send_data(sender, source_face_bytes, metadata, modules.globals.push_addr)
    finally:
        SOURCE_UPLOAD_LOCK.release()

def send_temp_frame(temp_face: Frame)-> None:
    sender = push_socket(modules.globals.push_addr_two)
    source_face_bytes = temp_face.tobytes()
    metadata = {
            'manyface':(modules.globals.many_faces),
            'dtype_temp':str(temp_face.dtype),
            'shape_temp':temp_face.shape,
            
            #'shape_temp':temp_frame.shape
        }
    send_data(sender, source_face_bytes, metadata,modules.globals.push_addr) 

def receive_processed_frame(output_queue: queue.Queue) -> None:
    """
    Pull one processed frame sent by the Colab server.

    @param {queue.Queue} output_queue - queue that receives the decoded array.
    @returns {None} the frame is delivered through the queue.
    """
    pull_socket_ = None
    try:
        pull_socket_ = pull_socket(modules.globals.pull_addr)
        try:
            meta_data_json = pull_socket_.recv_json()
        except zmq.Again:
            print(f"[REMOTE] Timeout waiting for a processed frame on {modules.globals.pull_addr}.")
            return
        print(meta_data_json)
        total_chunk = meta_data_json['total_chunk']
        pull_socket_.send_string("ACK")
        source_array_bytes = b''
        with tqdm(total=total_chunk, desc="Receiving chunks", unit="chunk") as pbar:
            for i in range(total_chunk):
                chunk = pull_socket_.recv()
                source_array_bytes += chunk
                pull_socket_.send_string(f"ACK {i + 1}/{total_chunk}")
                pbar.set_postfix_str(f'Chunk {i + 1}/{total_chunk}')
                pbar.update(1)

        end_message = pull_socket_.recv()
        if end_message == b"END":
            pull_socket_.send_string("Final ACK")

        source_array = np.frombuffer(source_array_bytes, dtype=np.dtype(meta_data_json['dtype_source'])).reshape(meta_data_json['shape_source'])

        output_queue.put(source_array)
    except zmq.Again:
        print(f"[REMOTE] Timeout while receiving a processed frame on {modules.globals.pull_addr}.")
    except zmq.ZMQError as error:
        print(f"[REMOTE] Socket error while receiving a processed frame: {error}")
    finally:
        if pull_socket_ is not None:
            pull_socket_.close()
def send_streams(cap: VideoCapture) -> subprocess.Popen[bytes]:
    
    ffmpeg_command = [
        'ffmpeg',
        '-fflags', 'nobuffer',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s',  f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        '-r',  '30',#str(int(cap.get(cv2.CAP_PROP_FPS))),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        
        #'-flags', 'low_delay',
        #'-rtbufsize', '100M',
        '-g', modules.globals.gof,#'15',  # GOP size set to 1 to make each frame a keyframe
        '-b:v', modules.globals.bitrate,#'800k',  # Set bitrate to 1 Mbps
        '-maxrate', modules.globals.maxrate,#'800k',  
        '-bufsize', modules.globals.bufsize,#'800k',
        '-f', 'mpegts', modules.globals.push_addr_two #'tcp://127.0.0.1:5552'
    ]


    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    return ffmpeg_process
def recieve_streams(cap: VideoCapture)->subprocess.Popen[bytes]:
    """
    Start the ffmpeg process that pulls the processed stream from the Colab tunnel.

    @param {VideoCapture} cap - local capture handle, kept for call-site compatibility.
    @returns {subprocess.Popen} ffmpeg process exposing raw BGR frames on stdout.
    """
    ffmpeg_command_recie = [
    'ffmpeg',
    '-flags','low_delay',
    '-analyzeduration','0',
    '-probesize','32',
    '-i',modules.globals.pull_addr,
    '-f','rawvideo',
    '-pix_fmt','bgr24',
    '-s','960x540',
    'pipe:1'
    ]
    
    ffmpeg_process_com = subprocess.Popen(ffmpeg_command_recie, stdout=subprocess.PIPE)
    return ffmpeg_process_com

def write_to_stdin(temp_frame: Frame, queue: queue.Queue, stream_out: subprocess.Popen) -> bool:
    """
    Push one raw BGR frame into the ffmpeg streaming process.

    @param {Frame} temp_frame - frame captured from the local camera.
    @param {queue.Queue} queue - kept for call-site compatibility, unused here.
    @param {subprocess.Popen} stream_out - ffmpeg push process with stdin piped.
    @returns {bool} True when the frame was written, False when the pipe is gone.
    """
    if stream_out is None or stream_out.stdin is None:
        return False
    try:
        stream_out.stdin.write(temp_frame.tobytes())
        stream_out.stdin.flush()
        return True
    except (OSError, ValueError):
        return False


def read_exact(stream: Any, size: int) -> bytes:
    """
    Read exactly `size` bytes from a binary pipe.

    @param {Any} stream - binary stream, usually ffmpeg stdout.
    @param {int} size - number of bytes required.
    @returns {bytes} buffered bytes, or None when the stream ended early.
    """
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            return None
        buffer.extend(chunk)
    return bytes(buffer)


def read_from_stdout(stream_in: subprocess.Popen, output_queue: queue.Queue) -> bool:
    """
    Read one processed frame from the ffmpeg receiving process.

    @param {subprocess.Popen} stream_in - ffmpeg pull process with stdout piped.
    @param {queue.Queue} output_queue - queue that receives the decoded frame.
    @returns {bool} True when a frame was queued, False when the stream ended.
    """
    if stream_in is None or stream_in.stdout is None:
        return False
    raw_frame = read_exact(stream_in.stdout, STREAM_FRAME_SIZE)
    if raw_frame is None:
        return False
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((STREAM_HEIGHT, STREAM_WIDTH, STREAM_CHANNELS))
    output_queue.put(frame)
    return True
def swap_face_remote(temp_frame: Frame,stream_out:subprocess.Popen[bytes],stream_in: subprocess.Popen[bytes]) -> Frame:
    """
    Process a single frame through the remote ffmpeg pipeline.

    @param {Frame} temp_frame - frame to be processed remotely.
    @param {subprocess.Popen} stream_out - ffmpeg push process.
    @param {subprocess.Popen} stream_in - ffmpeg pull process.
    @returns {Frame} processed frame, or the untouched frame when the pipeline fails.
    """
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    def push_frames() -> None:
        while True:
            queued_frame = input_queue.get()
            if queued_frame is None:
                break
            if not write_to_stdin(queued_frame, input_queue, stream_out):
                break

    def pull_frame() -> None:
        read_from_stdout(stream_in, output_queue)

    write_thread = threading.Thread(target=push_frames, daemon=True)
    read_thread = threading.Thread(target=pull_frame, daemon=True)

    read_thread.start()
    write_thread.start()

    processed_frame = temp_frame
    input_queue.put(temp_frame)
    try:
        processed_frame = output_queue.get(timeout=30)
    except queue.Empty:
        processed_frame = temp_frame
    finally:
        input_queue.put(None)
        write_thread.join(timeout=5)
        read_thread.join(timeout=5)

    return processed_frame
    
    
def swap_frame_face_remote(source_frame: Frame,temp_frame: Frame) -> Frame:
    #input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # Start threads for stdin and stdout
    write_thread = threading.Thread(target=send_source_frame, args=(source_frame,))
    write_thread_tw = threading.Thread(target=send_temp_frame, args=(temp_frame,))
    read_thread_ = threading.Thread(target=receive_processed_frame, args=(output_queue,))
    
    write_thread.start()
    write_thread_tw.start()
    read_thread_.start()
    
    # Send the frame to the stdin thread
    
    # Wait for the processed frame from the stdout thread
    processed_frame = temp_frame
    try:
        processed_frame = output_queue.get(timeout=RECEIVE_TIMEOUT_S)
    except queue.Empty:
        print("[REMOTE] No processed frame received, returning the original frame.")
    
    # Stop the threads
    write_thread.join(timeout=25)
    write_thread_tw.join(timeout=25)
    read_thread_.join(timeout=25)

    return processed_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    source_frame = cv2.imread(source_path)
    frames = []
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        frames.append(temp_frame)
        ''' 
        try:
            result = process_frame(source_frame, temp_frame)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(exception)
            pass'''
        if progress:
            progress.update(1)
    # Convert list of frames into a 4D NumPy array
    temp_frame = np.stack(frames, axis=0)
    print("Video Dimentsion",temp_frame.ndim,temp_frame.shape)
    result = process_frame(source_frame, temp_frame)
    print("Video Dimension result",result.shape)
    for i,frame in enumerate(result):
        print(f"write to {temp_frame_paths[i]}" )
        cv2.imwrite(temp_frame_paths[i], frame)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_frame = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
    result = process_frame(source_frame, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    #modules.processors.frame.core.remote_process_video(source_path, temp_frame_paths, process_frames)
    process_frames(source_path, temp_frame_paths)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import subprocess
import time
import psutil
import pyautogui
import vlc
import threading
import pygetwindow as gw
import json
import wave
import contextlib
from datetime import datetime, timedelta
import sys
from screeninfo import get_monitors
import numpy as np
import pyaudio
from scipy.fft import fft
from queue import Queue,Empty
import cv2

def get_video_fps_opencv(file_path):
    """OpenCV ile videonun FPS bilgisini al."""
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

placeholder_image_path = None
current_file_type = None

# Function that allows you to get the file path when working with PyInstaller
def resource_path(relative_path):
    """PyInstaller ile çalışırken dosya yolunu al."""
    try:
        base_path = sys._MEIPASS  # PyInstaller'ın kullandığı geçici dizin
    except Exception:
        base_path = os.path.abspath(".")  # Geliştirme aşamasında, mevcut dizin
    return os.path.join(base_path, relative_path)

# Button creation function creates a button with the specified image
def create_image_button(root, image_path, size=(80, 80), command=None):
    # Dinamik dosya yolunu kullanarak resmi aç
    img = Image.open(resource_path(image_path))
    img = img.resize(size, Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    button = tk.Button(root, image=photo, borderwidth=0, highlightthickness=0, command=command)
    button.image = photo  # Referansı saklamak, resmin gösterilmesi için gerekli
    return button

def create_popup_menu(event, tree):
    popup_menu = tk.Menu(tree, tearoff=0)
    popup_menu.add_command(label="Add", command=lambda: add_file_to_table(tree))
    popup_menu.add_command(label="Remove", command=lambda: remove_item(tree))
    popup_menu.post(event.x_root, event.y_root)

# Global variable to store the dragged item information
drag_data = {"widget": None, "item_id": None, "start_index": None}

def on_drag_start(event, tree):
    """Save information when starting dragging."""
    drag_data["widget"] = event.widget
    drag_data["item_id"] = tree.identify_row(event.y)  # Sürüklenen öğe ID'si
    if drag_data["item_id"]:
        drag_data["start_index"] = tree.index(drag_data["item_id"])

def on_drag_motion(event, tree):
    """No action is required while dragging."""
    pass

def on_drag_end(event, tree):
    """Swap two items when drag-and-drop operation is complete."""
    drop_item_id = tree.identify_row(event.y)  # Dropped item ID
    if drop_item_id and drag_data["item_id"] and drop_item_id != drag_data["item_id"]:
       # Swap the dropped item with the dragged item
        start_index = drag_data["start_index"]
        end_index = tree.index(drop_item_id)

        if start_index != end_index:
            # Get both items
            start_values = tree.item(drag_data["item_id"], 'values')
            end_values = tree.item(drop_item_id, 'values')

            # Change the data of the items
            tree.item(drag_data["item_id"], values=end_values)
            tree.item(drop_item_id, values=start_values)

            # Print when list is updated
            print_list(tree)

    # Reset drag data
    drag_data["widget"] = None
    drag_data["item_id"] = None
    drag_data["start_index"] = None

def print_list(tree):
    """Print the current version of the list."""
    print("Updated list:")
    items = tree.get_children()
    for item in items:
        values = tree.item(item, 'values')
        print(f"Time: {values[0]}, Description: {values[1]}, Path: {values[2]}")

# Function to remove selected item from table
def remove_item(tree):
    selected_item = tree.selection()
    if selected_item:
        tree.delete(selected_item)
    else:
        messagebox.showwarning("Warning", "Select an item to remove.")


# Menu creation function
def create_menu(root, tree):
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # Creating a file menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="New", command=lambda: clear_table(tree))  # Clear the table with New
    file_menu.add_command(label="Open", command=lambda: load_table_from_file(tree))  # Load from file with Open
    file_menu.add_command(label="Save", command=lambda: save_table_to_file(tree))  # Save the table with Save
    file_menu.add_command(label="Save As", command=lambda: save_table_to_file(tree))  # Same function for Save As function
    file_menu.add_separator()
    file_menu.add_command(label="Quit", command=root.quit)

    # Options menu
    options_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Options", menu=options_menu)
    options_menu.add_command(label="Preferences")

    # About menu
    about_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="About", menu=about_menu)
    about_menu.add_command(label="VSA Player", command=lambda: messagebox.showinfo("About", "VSA Player"))

    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="User Guide", command=lambda: messagebox.showinfo("Help", "User Guide Not Available"))


def create_video_area(main_frame, vlc_instance):
    video_frame = tk.Frame(main_frame)
    video_frame.pack(side=tk.RIGHT, padx=5, pady=5, expand=True)

    video_label = tk.Label(video_frame, text="Video Preview")
    video_label.pack()

    canvas = tk.Canvas(video_frame, width=450, height=300, bg="black")
    canvas.pack()

    # VLC Media Player in Canvas
    player_preview = vlc_instance.media_player_new()
    player_preview.set_hwnd(canvas.winfo_id())
    #player_preview.audio_set_mute(True)  # Preview'da ses kapalı

    # VLC Media Player to run on external display
    player_external = vlc_instance.media_player_new()
    
    # Get monitor information
    monitors = get_monitors()
    if len(monitors) > 1:
        # If there is a second monitor, play full screen on the second monitor
        monitor = monitors[1]
    else:
        # If there is only one monitor, play on main screen
        monitor = monitors[0]
    
    # Create a new window for external player
    external_window = tk.Toplevel()
    external_window.geometry(f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}")
    external_window.overrideredirect(True)  # Remove window decorations
    
    external_canvas = tk.Canvas(external_window, width=monitor.width, height=monitor.height, bg="black")
    external_canvas.pack(fill=tk.BOTH, expand=True)
    
    player_external.set_hwnd(external_canvas.winfo_id())
    player_external.set_fullscreen(True)

    return canvas, video_frame, player_preview, player_external, external_window,external_canvas

# Save the items in the list to a JSON file
def save_table_to_file(tree):
    # We get all elements in the table
    items = tree.get_children()
    data = []

    for item in items:
        # We get the information of each item
        item_data = {
            "time": tree.item(item, 'values')[0],
            "description": tree.item(item, 'values')[1],
            "file_path": tree.item(item, 'values')[2],
        }
        data.append(item_data)

    # We select the file and specify the location to save it
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    
    if file_path:
        # We write to the JSON file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        messagebox.showinfo("Information", "Table saved successfully.")

# Loading data from JSON file into table
def load_table_from_file(tree):
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    
    if file_path:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Clear the current table
        for item in tree.get_children():
            tree.delete(item)
        
        # Re-add the saved data to the table
        for item_data in data:
            tree.insert('', tk.END, text='▶', values=(item_data['time'], item_data['description'], item_data['file_path']))
        
        messagebox.showinfo("Information", "Table loaded successfully.")


# Function to clear all elements in the table
def clear_table(tree):
    tree.delete(*tree.get_children())
    messagebox.showinfo("Information", "Table cleared. You can start adding new items.")

# Function to start the VSA program
def run_vsa(vsa_path):
    command = [vsa_path, "/minimize"]
    print("Starting VSA without loading a project...")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Loop to check if VSA is working
    while True:
        if "VSA.exe" in (p.name() for p in psutil.process_iter(attrs=['name'])):
            print("VSA started.")
            break
        time.sleep(0.1)
    return process

####### DETECT TONE ##########

# Tone detection parameters
SAMPLE_RATE = 44100
TONE_FREQ = 22000
CHUNK_SIZE_MAX = int(SAMPLE_RATE * 0.025)
FREQUENCY_THRESHOLD = 100

def detect_tone(data):
    fft_data = fft(data)
    freqs = np.fft.fftfreq(len(fft_data), 1 / SAMPLE_RATE)
    peak_freq = abs(freqs[np.argmax(np.abs(fft_data))])
    return peak_freq

def tone_detection_thread(stop_event, time_queue):
    p = pyaudio.PyAudio()
    device_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "CABLE" in info['name'] and info['maxInputChannels'] > 0:
            device_index = i
            print(f"Using input device: {info['name']} (Index: {device_index})")
            break
    if device_index is None:
        print("No suitable VB-Cable input device found. Exiting tone detection.")
        return

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK_SIZE_MAX)

    print("Listening for 22 kHz tone...")

    while not stop_event.is_set():
        try:
            audio_data = np.frombuffer(stream.read(CHUNK_SIZE_MAX, exception_on_overflow=False), dtype=np.int16)
            detected_freq = detect_tone(audio_data)
            if abs(detected_freq - TONE_FREQ) < FREQUENCY_THRESHOLD:
                detected_time = datetime.now()
                start_time = detected_time - timedelta(milliseconds=20)
                print(f"Tone detected! Start time: {start_time.strftime('%H:%M:%S.%f')[:-3]}")
                time_queue.put(start_time)
        except Exception as e:
            print(f"Error in tone detection: {e}")

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Tone detection terminated.")
####### DETECT TONE ##########

def open_project(file_path,time_queue):
    time.sleep(2)
    # Start tone detection thread
    stop_event = threading.Event()
    tone_thread = threading.Thread(target=tone_detection_thread, args=(stop_event, time_queue))
    tone_thread.start()

    vsa_window = None
    for window in gw.getAllWindows():
        if window.title.startswith("VSA"):
            vsa_window = window
            break

    if vsa_window:
        if vsa_window.isMinimized:
            vsa_window.restore()
        vsa_window.activate()
        print("VSA window found.")
        time.sleep(1)
    else:
        print("VSA window not found.")
        return

    print("Sending Ctrl + O to open a new project...")
    pyautogui.hotkey('ctrl', 'o')
    time.sleep(1)
    
    normalized_file_path = os.path.normpath(file_path)
    print(f"Entering project path: {normalized_file_path}")
    pyautogui.write(normalized_file_path)
    
    pyautogui.press('enter')
    time.sleep(4)
    pyautogui.press('enter')

    vsa_start_time = datetime.now()
    print(f"VSA Current Time: {vsa_start_time}")

    time.sleep(0.5)
    pyautogui.hotkey('alt', 'space')
    time.sleep(0.5)
    pyautogui.press('n')
    print("VSA minimized.")

    print("VSA is started")

    # Stop tone detection thread
    stop_event.set()
    tone_thread.join()

    return vsa_start_time, vsa_window

# Placeholder image selection function
def set_placeholder_image():
    global placeholder_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
    if file_path:
        placeholder_image_path = file_path
        messagebox.showinfo("Success", f"Placeholder image set to: {file_path}")
        print("Placeholder image set to: {file_path}")

# A function to get the length of the WAV file in seconds
def get_wav_duration(file_path):
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error: Unable to retrieve the duration of the WAV file: {e}")
        return None
    
# Function to add to the table by selecting a file
def add_file_to_table(tree):
    # Specify the folder to open by default
    default_dir = r"C:\VSA_Video"  # Here a directory under C:\ folder is selected
    
    # Selecting multiple files
    file_paths = filedialog.askopenfilenames(initialdir=default_dir, filetypes=[("VSA/WAV", "*.vsa *.wav")])
    
    if file_paths:
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            description = file_name
            time = "00:00:00"

            # Get duration if WAV file
            if file_path.endswith('.wav'):
                duration = get_wav_duration(file_path)
                if duration is not None:
                    minutes, seconds = divmod(duration, 60)
                    hours, minutes = divmod(minutes, 60)
                    time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

            # If it is a VSA file, check the WAV file with the same name
            elif file_path.endswith('.vsa'):
                wav_file_path = file_path.replace('.vsa', '.wav')
                print(wav_file_path)
                if os.path.exists(wav_file_path):
                    duration = get_wav_duration(wav_file_path)
                    print(duration)
                    if duration is not None:
                        minutes, seconds = divmod(duration, 60)
                        hours, minutes = divmod(minutes, 60)
                        time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

            tree.insert('', tk.END, text='▶', values=(time, description, file_path, "✓"))

    
    # Print when list is updated
    print_list(tree)

def format_time(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    milliseconds = milliseconds % 1000  # Remaining milliseconds
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

def update_runtime(player_preview, run_time_entry):
    last_time_str = None  # A variable to hold the last written time

    while True:
        if player_preview.is_playing():
            current_time_ms = player_preview.get_time()  # Instantaneous time (ms)
            total_time_ms = player_preview.get_length()  # Total time (ms)

            # Format new duration as MM:SS:MS
            current_time_str = f"{format_time(current_time_ms)} / {format_time(total_time_ms)}"

            # If the duration is the same as before, do not write
            if current_time_str != last_time_str:
                run_time_entry.delete(0, tk.END)
                run_time_entry.insert(0, current_time_str)
                last_time_str = current_time_str  # Update the last written time

        time.sleep(0.00000001)  # Update every millisecond



def play_media(file_path, player_preview, player_external, vlc_instance, canvas, external_canvas, external_window, sync_param_entry):
    vsa_window = None
    media_file = None
    print(file_path)
    if file_path.endswith('.vsa'):
        print("vsa ya girdi")
        # If VSA file is playing, mute internal and external media audio
        print("VSA project detected, muting internal and external media.")
        player_preview.audio_set_mute(True)  # Internal sound off
        player_external.audio_set_mute(True)  # External sound off

        

        # Start tone detection thread
        stop_event = threading.Event()
        time_queue = Queue()
        tone_thread = threading.Thread(target=tone_detection_thread, args=(stop_event, time_queue))
        tone_thread.start()

        # Open VSA project
        vsa_start_time, vsa_window = open_project(file_path,time_queue)
        if vsa_start_time is None:
            print("VSA project could not be opened.")
            stop_event.set()
            tone_thread.join()
            return

        # Find the MP4 or WAV file associated with the VSA file
        mp4_file_path = file_path.replace('.vsa', '.mp4')
        wav_file_path = file_path.replace('.vsa', '.wav')
        
        if os.path.exists(mp4_file_path):
            media_file = mp4_file_path
            print("media file :", media_file)
        elif os.path.exists(wav_file_path):
            media_file = wav_file_path
            # Show placeholder image if WAV file is playing
            if placeholder_image_path:
                
                img = Image.open(placeholder_image_path)

                monitor_width = external_window.winfo_screenwidth()
                monitor_height = external_window.winfo_screenheight()

                img_external = img.resize((monitor_width, monitor_height), Image.Resampling.LANCZOS)
                photo_external = ImageTk.PhotoImage(img_external)

                external_canvas.create_image(0, 0, image=photo_external, anchor=tk.NW)
                external_canvas.image = photo_external  

                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()

                img_internal = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                photo_internal = ImageTk.PhotoImage(img_internal)

                canvas.create_image(0, 0, image=photo_internal, anchor=tk.NW)
                canvas.image = photo_internal  

        else:
            print(f"No corresponding MP4 or WAV file found for: {file_path}")
            stop_event.set()
            tone_thread.join()
            return
        
        
    else:
        print("wava girdi")
        media_file = file_path
        # If only WAV file is playing, we will get the sound from internal media
        print("WAV file detected, enabling internal media sound only.")
        #player_preview.audio_set_mute(False)  
        player_external.audio_set_mute(False) 

        # Show placeholder image if WAV file is playing
        if placeholder_image_path:
            img = Image.open(placeholder_image_path)

            monitor_width = external_window.winfo_screenwidth()
            monitor_height = external_window.winfo_screenheight()

            img_external = img.resize((monitor_width, monitor_height), Image.Resampling.LANCZOS)
            photo_external = ImageTk.PhotoImage(img_external)

            external_canvas.create_image(0, 0, image=photo_external, anchor=tk.NW)
            external_canvas.image = photo_external  

            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            img_internal = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            photo_internal = ImageTk.PhotoImage(img_internal)

            canvas.create_image(0, 0, image=photo_internal, anchor=tk.NW)
            canvas.image = photo_internal  

    # Load and play media file
    media = vlc_instance.media_new(media_file)
    player_preview.set_media(media)
    player_external.set_media(media)

    # OpenCV ile FPS bilgisi al
    # try:
    #     fps = get_video_fps_opencv(media_file)
    #     print(f"Video FPS (OpenCV ile): {fps}")
    # except Exception as e:
    #     print(f"Unable to retrieve FPS with OpenCV: {e}")

    # # Video 30 FPS hızında oynatılacak
    # player_preview.set_rate(30.0 / fps)
    # player_external.set_rate(30.0 / fps)

    # Start media
    player_preview.play()
    player_external.play()
    video_start_time = datetime.now()

    
    print(f"Video Star Time: {video_start_time}")

    try:
        sync_time = float(sync_param_entry.get())
    except ValueError:
        sync_time = 0  # If an invalid value is entered, default to 0

    
    if file_path.endswith('.vsa'): 
        # Tone detection zamanını al ve yazdır
        
        tone_start_time = time_queue.get()  # 10 saniye bekle
        print(f"Toneeeeeeeeeeee: {tone_start_time}")
        time_difference = (video_start_time - tone_start_time).total_seconds()
        print("Time Difference:", time_difference)

        # Use sync_time in both conditions
        if time_difference > 0:
            # Video started later, add sync_time and fast forward
            total_time_difference = time_difference + sync_time
            print(f"Video started later. Fast-forwarding by {total_time_difference} seconds.")
            player_preview.set_time(int(total_time_difference * 1000))  # Set time in milliseconds
            player_external.set_time(int(total_time_difference * 1000))  # Set time in milliseconds
        elif time_difference < 0:
            # Video started earlier, add sync_time and set the delay
            total_time_difference = abs(time_difference) + sync_time
            print(f"Video started earlier. Pausing for {total_time_difference} seconds.")
            time.sleep(total_time_difference)

    # Control until media is finished
    while True:
        state_preview = player_preview.get_state()
        state_external = player_external.get_state()
        if state_preview == vlc.State.Ended and state_external == vlc.State.Ended:
            break
        time.sleep(0.1)
    
    # At the end of the function, stop the tone detection thread
    if 'stop_event' in locals():
        stop_event.set()
        tone_thread.join()

    print("Playback finished")



def play_media_thread(file_path, player_preview, player_external, vlc_instance, canvas, external_canvas, external_window, sync_param_entry):
    thread = threading.Thread(target=play_media, args=(file_path, player_preview, player_external, vlc_instance, canvas, external_canvas, external_window, sync_param_entry),daemon=True)
    thread.start()
    return thread


def on_play_button(tree, player_preview, player_external, vlc_instance, controls, now_playing_entry, run_time_entry, external_window, canvas,external_canvas,sync_param_entry):
    global current_file_type  # Declare it global to track the file type

    # If media is paused, resume it
    if controls["paused"]:
        if current_file_type == 'vsa':  # Only restore VSA if the current file is a VSA project
            vsa_window = get_vsa_window()
            if vsa_window:
                if vsa_window.isMinimized:
                    print("Restoring VSA window.")
                    vsa_window.restore()
                    time.sleep(0.5)
                vsa_window.activate()
                time.sleep(0.5)  # Wait for the VSA window to activate
                print("Resuming VSA project.")
                pyautogui.press('enter')  # Send 'Enter' to resume VSA

        # Resume media playback for both VSA and WAV files
        player_preview.play()
        player_external.play()
        controls["paused"] = False
        print("Resuming media from pause.")
        return

    # Get the selected item from the tree
    selected_item = tree.selection()[0] if tree.selection() else None
    items = tree.get_children()

    # If no item is selected, start from the first item
    start_index = tree.index(selected_item) if selected_item else 0
    items_to_play = items[start_index:]

    def play_all_items():
        global current_file_type  # Make sure we update the global variable here
        for item in items_to_play:
            values = tree.item(item, 'values')
            file_path = values[2]

            # Set the currently playing file type (VSA or WAV)
            current_file_type = 'vsa' if file_path.endswith('.vsa') else 'wav'

            # Update the "Now Playing" field in the GUI
            now_playing_entry.delete(0, tk.END)
            now_playing_entry.insert(0, os.path.basename(file_path))

            # Select and scroll to the current item in the tree
            tree.selection_set(item)
            tree.see(item)
            tree.update()

            print(f"Playing file: {file_path}, current file type: {current_file_type}")

            # Start the media playback in a separate thread
            thread = play_media_thread(file_path, player_preview, player_external, vlc_instance, canvas, external_canvas, external_window, sync_param_entry)
            threading.Thread(target=update_runtime, args=(player_preview, run_time_entry),daemon=True).start()
            thread.join()  # Wait for the current media to finish before moving to the next one

    external_window.deiconify()
    threading.Thread(target=play_all_items,daemon=True).start()



def get_vsa_window():
    # Tüm pencereleri kontrol et ve VSA penceresini bul
    for window in gw.getAllWindows():
        if window.title.startswith("VSA"):
            return window
    return None

def on_pause_button(controls, player_preview, player_external):
    global current_file_type  # Access the file type
    print("current_file_type",current_file_type)

    if not controls["paused"]:
        if current_file_type == 'vsa':
            # If the current file is a VSA project, pause it
            vsa_window = get_vsa_window()
            if vsa_window:
                if vsa_window.isMinimized:
                    vsa_window.restore()
                    time.sleep(0.5)
                vsa_window.activate()
                time.sleep(0.5)
                pyautogui.press('space')  # Send space to VSA to pause
                print("Pausing VSA project.")
        else:
            # If it's a WAV file, pause media player only
            print("Pausing WAV file.")
        
        player_preview.pause()
        player_external.pause()
        video_pause_time = datetime.now()
        print(f"Video Pause Time: {video_pause_time}")
        controls["paused"] = True
        print("Media paused.")
    else:
        print("Media is already paused.")




def on_stop_button(controls, player_preview, player_external, external_window):
    player_preview.stop()
    player_external.stop()
    controls["paused"] = False
    controls["playing"] = False
    external_window.withdraw()  
    print("Media stopped.")


# Main GUI creation function
def create_gui():
    root = tk.Tk()
    root.title("Video Player")

    root.resizable(False, False)

    controls = {"cap": None, "playing": False, "paused": False, "playing_event": threading.Event()}

    main_frame = tk.Frame(root)
    main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    vlc_instance = vlc.Instance()

    # Info frame
    info_frame = tk.Frame(main_frame)
    info_frame.pack(fill=tk.X, pady=10)

    now_playing_label = tk.Label(info_frame, text="Now Playing:", font=('Arial', 12))
    now_playing_label.pack(side=tk.LEFT, padx=5)

    now_playing_entry = tk.Entry(info_frame, width=40, font=('Arial', 12))
    now_playing_entry.pack(side=tk.LEFT)

    run_time_label = tk.Label(info_frame, text="Run Time:", font=('Arial', 12))
    run_time_label.pack(side=tk.LEFT, padx=10)

    run_time_entry = tk.Entry(info_frame, width=15, font=('Arial', 12))
    run_time_entry.pack(side=tk.LEFT)

    # Add it to the create_gui function
    sync_param_label = tk.Label(info_frame, text="Sync Param:", font=('Arial', 12))
    sync_param_label.pack(side=tk.LEFT, padx=10)

    sync_param_entry = tk.Entry(info_frame, width=5, font=('Arial', 12))
    sync_param_entry.pack(side=tk.LEFT)
    sync_param_entry.insert(0, "0")  # Set the default value to 0

    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)

    bottom_frame = tk.Frame(main_frame)
    bottom_frame.pack(fill=tk.BOTH, expand=True)

    table_frame = tk.Frame(bottom_frame)
    table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Columns and headers for TreeView (table)
    columns = ('Time', 'Description', 'File Location / Event Properties')
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)

    tree.heading('#0', text='', anchor=tk.CENTER)
    tree.heading('Time', text='Time', anchor=tk.W)
    tree.heading('Description', text='Description', anchor=tk.W)
    tree.heading('File Location / Event Properties', text='File Location / Event Properties', anchor=tk.W)

    tree.column('#0', width=50, stretch=tk.NO, minwidth=50, anchor=tk.CENTER)
    tree.column('Time', width=100, stretch=tk.NO, minwidth=100)
    tree.column('Description', width=200, stretch=tk.NO, minwidth=200)
    tree.column('File Location / Event Properties', width=400, stretch=tk.NO, minwidth=400)

    tree.pack(fill=tk.BOTH, expand=True)

    # Right click to open popup menu
    tree.bind("<Button-3>", lambda event: create_popup_menu(event, tree))

    # Drag and drop event
    tree.bind("<Button-1>", lambda event: on_drag_start(event, tree))
    tree.bind("<B1-Motion>", lambda event: on_drag_motion(event, tree))
    tree.bind("<ButtonRelease-1>", lambda event: on_drag_end(event, tree))

    # Creating the menu, sending the tree parameter
    create_menu(root, tree)

    # Creating a video preview area
    canvas, video_frame, player_preview, player_external, external_window,external_canvas = create_video_area(bottom_frame, vlc_instance)

    # Placeholder button is moved to the top of the video preview
    placeholder_button = tk.Button(video_frame, text="Set Placeholder", command=set_placeholder_image, width=15, 
                                   height=3, font=('Times New Roman', 10),bd=2)
    placeholder_button.pack(side=tk.TOP, padx=10, pady=10)

    # Creating and placing buttons
    button_images = ['add.png', 'play.png', 'pause.png', 'stop.png']
    button_commands = [
        lambda: add_file_to_table(tree),
        lambda: on_play_button(tree, player_preview, player_external, vlc_instance, controls, now_playing_entry, run_time_entry, external_window, canvas,external_canvas,sync_param_entry),
        lambda: on_pause_button(controls, player_preview, player_external),
        lambda: on_stop_button(controls, player_preview, player_external, external_window)
    ]   

    for img, cmd in zip(button_images, button_commands):
        button = create_image_button(button_frame, os.path.join('photo', img), command=cmd)
        button.pack(side=tk.LEFT, padx=10)

    # Starting VSA, project will be loaded
    vsa_exe_path = r"C:\Program Files (x86)\Brookshire Software\Visual Show Automation Professional\VSA.exe"
    run_vsa(vsa_exe_path)

    root.mainloop()

# Main function of the program
if __name__ == "__main__":
    create_gui()
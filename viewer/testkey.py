from pynput import keyboard

import hl2ss_lnm
import hl2ss

def on_press(key):
    global running
    global enable
    global listening
    if key == keyboard.Key.esc:
        # Stop the loop
        running = False
        enable = False
        listening = False
        print("Esc pressed")
        return False
    return True

host = '192.168.1.185'

listener = keyboard.Listener(on_press=on_press)
listener.start()
listening = True

while (listening):
    print("Waiting for data...")
    # listening = False
print("Received bounding box")
listener.join()


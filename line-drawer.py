import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

class RectangularAreaSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        self.image = self.original_image.copy()
        self.areas = []
        
        # Drawing variables
        self.drawing = False
        self.start_point = None
        self.end_point = None

    def create_window(self):
        cv2.namedWindow('Select Restricted Areas')
        cv2.setMouseCallback('Select Restricted Areas', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        # Reset image to original for clean drawing
        self.image = self.original_image.copy()

        # Draw existing areas
        for area in self.areas:
            cv2.rectangle(self.image, 
                          (area['coordinates'][0], area['coordinates'][1]), 
                          (area['coordinates'][2], area['coordinates'][3]), 
                          (0, 255, 0), 2)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Draw current rectangle while dragging
                cv2.rectangle(self.image, self.start_point, (x, y), (255, 0, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # Ensure rectangle is created in a consistent way (top-left to bottom-right)
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            # Prompt for area name
            name = simpledialog.askstring("Area Name", "Enter name for this restricted area:")
            if name:
                self.areas.append({
                    "name": name,
                    "coordinates": [x1, y1, x2, y2]
                })

        cv2.imshow('Select Restricted Areas', self.image)

    def run(self):
        # Create Tkinter root for dialogs
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window

        self.create_window()

        while True:
            cv2.imshow('Select Restricted Areas', self.image)
            key = cv2.waitKey(1) & 0xFF

            # Press 'd' to delete last area
            if key == ord('d'):
                if self.areas:
                    self.areas.pop()
                    # Redraw image without last area
                    self.image = self.original_image.copy()
                    for area in self.areas:
                        cv2.rectangle(self.image, 
                                      (area['coordinates'][0], area['coordinates'][1]), 
                                      (area['coordinates'][2], area['coordinates'][3]), 
                                      (0, 255, 0), 2)
                    cv2.imshow('Select Restricted Areas', self.image)

            # Press 's' to save areas
            if key == ord('s'):
                self.save_areas()
            
            # ESC or 'q' to quit
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()

    def save_areas(self):
        if self.areas:
            with open('restricted_areas.json', 'w') as f:
                json.dump(self.areas, f, indent=4)
            messagebox.showinfo("Success", f"{len(self.areas)} restricted areas saved to restricted_areas.json")
        else:
            messagebox.showwarning("Warning", "No areas selected")

def select_image():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Select Reference Image", 
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if image_path:
        selector = RectangularAreaSelector(image_path)
        selector.run()

if __name__ == "__main__":
    select_image()
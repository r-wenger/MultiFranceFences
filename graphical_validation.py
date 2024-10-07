import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd

class FenceValidatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fence Validator")

        self.setup_grid()
        self.create_widgets()

        self.dataset_path = ""
        self.image_index = 0
        self.image_files = []
        self.results = []
        self.results_file = "results.csv"
        self.yes_folder = "yes_images"
        self.patch_size = 256

        if not os.path.exists(self.yes_folder):
            os.makedirs(self.yes_folder)

    def setup_grid(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure([0, 1, 2], weight=1)

    def create_widgets(self):
        self.create_image_frame()
        self.create_labels()
        self.create_buttons()
        self.create_help_button()
        self.create_confirmation_label()

    def create_image_frame(self):
        self.image_frame = tk.Frame(self.root)
        self.image_frame.grid(row=0, column=0, columnspan=3, pady=10, padx=10, sticky='nsew')
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure([0, 1], weight=1)
        
        self.canvas_ortho = tk.Canvas(self.image_frame, bd=2, relief=tk.SUNKEN)
        self.canvas_reference = tk.Canvas(self.image_frame, bd=2, relief=tk.SUNKEN)
        self.canvas_ortho.grid(row=0, column=0, padx=10, sticky='nsew')
        self.canvas_reference.grid(row=0, column=1, padx=10, sticky='nsew')

    def create_labels(self):
        self.label_ortho = tk.Label(self.image_frame, text="Ortho", font=("Arial", 14))
        self.label_reference = tk.Label(self.image_frame, text="Reference", font=("Arial", 14))
        self.label_ortho.grid(row=1, column=0, sticky='nsew')
        self.label_reference.grid(row=1, column=1, sticky='nsew')

        self.question_label = tk.Label(self.root, text="Are there fences?", font=("Arial", 14))
        self.question_label.grid(row=1, column=0, columnspan=2, pady=10, sticky='nsew')

    def create_buttons(self):
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky='nsew')
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure([0, 1], weight=1)

        self.yes_button = tk.Button(self.button_frame, text="Yes", command=lambda: self.record_result("Yes"), width=10, height=2, bg='lightgreen')
        self.no_button = tk.Button(self.button_frame, text="No", command=lambda: self.record_result("No"), width=10, height=2, bg='salmon')
        self.yes_button.grid(row=0, column=0, padx=10, sticky='nsew')
        self.no_button.grid(row=0, column=1, padx=10, sticky='nsew')

        self.load_button = tk.Button(self.root, text="Load Dataset", command=self.load_dataset, font=("Arial", 12), bg='lightblue', width=20)
        self.load_button.grid(row=3, column=0, columnspan=2, pady=20, sticky='nsew')

    def create_help_button(self):
        self.help_button = tk.Button(self.root, text="?", command=self.show_help, font=("Arial", 12), width=2)
        self.help_button.grid(row=3, column=2, pady=20, padx=10, sticky='e')

    def create_confirmation_label(self):
        self.confirmation_label = tk.Label(self.root, text="", font=("Arial", 12), fg='red')
        self.confirmation_label.grid(row=4, column=0, columnspan=3, pady=10, sticky='nsew')

    def load_dataset(self):
        self.dataset_path = filedialog.askdirectory()
        if not self.dataset_path:
            return

        self.ortho_path = os.path.join(self.dataset_path, 'ortho')
        self.reference_path = os.path.join(self.dataset_path, 'fences_2m')
        if not (os.path.exists(self.ortho_path) and os.path.exists(self.reference_path)):
            messagebox.showerror("Error", "The selected folder must contain 'fences_2m' and 'reference' subfolders!")
            return

        self.image_files = sorted(f for f in os.listdir(self.ortho_path) if f.endswith('.tif'))
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the 'ortho' folder!")
            return

        self.load_results()
        self.show_image()

    def load_results(self):
        results_path = os.path.join(self.dataset_path, self.results_file)
        if os.path.exists(results_path):
            self.results = pd.read_csv(results_path).to_dict('records')
            self.image_index = len(self.results)
        else:
            self.results = []
            self.image_index = 0

    def show_image(self):
        if self.image_index >= len(self.image_files):
            messagebox.showinfo("Done", "All images processed!")
            self.confirmation_label.config(text="")
            return

        image_file = self.image_files[self.image_index]
        ortho_path = os.path.join(self.ortho_path, image_file)
        reference_path = os.path.join(self.reference_path, image_file)

        self.display_image(ortho_path, self.canvas_ortho, ortho=True)
        self.display_image(reference_path, self.canvas_reference, ortho=False)

    def display_image(self, image_path, canvas, ortho):
        try:
            img = Image.open(image_path)
            if ortho:
                img = img.convert('RGB')
            else:
                img = self.convert_binary_to_rgb(img)
            img = img.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            canvas.image = img_tk
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.config(scrollregion=canvas.bbox(tk.ALL))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def convert_binary_to_rgb(self, img):
        # Convert binary (0, 1) image to RGB
        arr = np.array(img)
        rgb_arr = np.stack([arr * 255] * 3, axis=-1)
        return Image.fromarray(rgb_arr.astype('uint8'), 'RGB')

    def record_result(self, result):
        if self.image_index >= len(self.image_files):
            return

        image_file = self.image_files[self.image_index]
        self.results.append({"filename": image_file, "result": result})
        self.save_results()

        if result == "Yes":
            self.save_yes_image(image_file)

        self.confirmation_label.config(text=f"{'Fences' if result == 'Yes' else 'No fences'} on the patch")

        self.image_index += 1
        self.show_image()

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.dataset_path, self.results_file), index=False)

    def save_yes_image(self, image_file):
        ortho_path = os.path.join(self.ortho_path, image_file)
        yes_path = os.path.join(self.yes_folder, image_file)
        try:
            img = Image.open(ortho_path)
            img.save(yes_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save 'Yes' image: {e}")

    def show_help(self):
        help_text = (
            "How to use the Fence Validator Application:\n\n"
            "1. Click 'Load Dataset' to select the main directory containing 'ortho' and 'reference' subfolders.\n"
            "2. The 'ortho' subfolder should contain orthophotos (RGB) and the 'reference' subfolder should contain binary reference images.\n"
            "3. The images will be displayed side by side: 'Ortho' on the left and 'Reference' on the right.\n"
            "4. Review the images and click 'Yes' if fences are present, otherwise click 'No'.\n"
            "5. Your responses will be saved immediately to 'results.csv' in the main directory.\n"
            "6. Images marked 'Yes' will be saved in the 'yes_images' folder for later review.\n"
            "7. You can close the application and resume later. The app will load your previous progress."
        )
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = FenceValidatorApp(root)
    root.mainloop()

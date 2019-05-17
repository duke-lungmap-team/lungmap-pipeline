import tkinter as tk
import ttkthemes as themed_tk
from tkinter import filedialog, ttk
import PIL.Image
import PIL.ImageTk
import json
import numpy as np
import lungmap_utils
from micap import utils as micap_utils

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2

BACKGROUND_COLOR = '#ededed'
BORDER_COLOR = '#bebebe'
HIGHLIGHT_COLOR = '#5294e2'

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 720

PAD_SMALL = 2
PAD_MEDIUM = 4

DEV_STAGES = [
    "E16.5",
    "E18.5",
    "P01",
    "P03",
    "P07"
]

MAG_VALUES = [
    "20X",
    "60X",
    "100X"
]

PROBES = lungmap_utils.client.get_probes()


class Application(tk.Frame):

    def __init__(self, master):

        tk.Frame.__init__(self, master=master)

        self.master.minsize(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self.master.config(bg=BACKGROUND_COLOR)
        self.master.title("LungMAP Region Generator")

        self.images = {}
        self.image_dims = None
        self.lm_query_top = None
        self.img_region_lut = None
        self.current_img = None
        self.tk_image = None

        self.current_dev_stage = tk.StringVar(self.master)
        self.current_mag = tk.StringVar(self.master)
        self.current_probe1 = tk.StringVar(self.master)
        self.current_probe2 = tk.StringVar(self.master)
        self.current_probe3 = tk.StringVar(self.master)

        self.dev_stage_option = None
        self.mag_option = None
        self.probe1_option = None
        self.probe2_option = None
        self.probe3_option = None
        self.query_results_list_box = None
        self.queried_images = {}
        self.download_progress_bar = None
        self.ref_img_name = None

        main_frame = tk.Frame(self.master, bg=BACKGROUND_COLOR)
        main_frame.pack(
            fill='both',
            expand=True,
            anchor='n',
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        file_chooser_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        file_chooser_frame.pack(
            fill=tk.X,
            expand=False,
            anchor=tk.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        bottom_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        bottom_frame.pack(
            fill='both',
            expand=True,
            anchor='n',
            padx=PAD_MEDIUM,
            pady=PAD_SMALL
        )

        file_chooser_button_frame = tk.Frame(
            file_chooser_frame,
            bg=BACKGROUND_COLOR
        )

        add_image_button = ttk.Button(
            file_chooser_button_frame,
            text='Load Images',
            command=self.query_lungmap_images
        )
        add_image_button.pack(side=tk.LEFT)

        preprocess_images_button = ttk.Button(
            file_chooser_button_frame,
            text='Pre-process Images',
            command=self.preprocess_images
        )
        preprocess_images_button.pack(side=tk.LEFT)

        save_regions_button = ttk.Button(
            file_chooser_button_frame,
            text='Save Regions',
            command=self.save_regions_json
        )
        save_regions_button.pack(side=tk.RIGHT, anchor=tk.N)

        self.current_label = tk.StringVar(self.master)

        self.label_option = ttk.Combobox(
            file_chooser_button_frame,
            textvariable=self.current_label,
            state='readonly'
        )
        self.label_option.bind('<<ComboboxSelected>>', self.select_label)
        self.label_option.pack(side=tk.RIGHT, fill='x', expand=False)

        file_chooser_button_frame.pack(
            anchor='n',
            fill='x',
            expand=False,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        file_list_frame = tk.Frame(
            file_chooser_frame,
            bg=BACKGROUND_COLOR,
            highlightcolor=HIGHLIGHT_COLOR,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1
        )
        file_scroll_bar = ttk.Scrollbar(file_list_frame, orient='vertical')
        self.file_list_box = tk.Listbox(
            file_list_frame,
            exportselection=False,
            height=4,
            yscrollcommand=file_scroll_bar.set,
            relief='flat',
            borderwidth=0,
            highlightthickness=0,
            selectbackground=HIGHLIGHT_COLOR,
            selectforeground='#ffffff'
        )
        self.file_list_box.bind('<<ListboxSelect>>', self.select_image)
        file_scroll_bar.config(command=self.file_list_box.yview)
        file_scroll_bar.pack(side='right', fill='y')
        self.file_list_box.pack(fill='x', expand=True)

        file_list_frame.pack(
            fill='x',
            expand=False,
            padx=PAD_MEDIUM,
            pady=PAD_SMALL
        )

        # TODO: put another toolbar here for image display options
        # including whether to view orig or corrected image, zoom options, etc.

        # the canvas frame's contents will use grid b/c of the double
        # scrollbar (they don't look right using pack), but the canvas itself
        # will be packed in its frame
        canvas_frame = tk.Frame(bottom_frame, bg=BACKGROUND_COLOR)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.pack(
            fill=tk.BOTH,
            expand=True,
            anchor=tk.N,
            side='right',
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        self.canvas = tk.Canvas(
            canvas_frame,
            cursor="tcross",
            takefocus=1
        )

        self.scrollbar_v = ttk.Scrollbar(
            canvas_frame,
            orient=tk.VERTICAL
        )
        self.scrollbar_h = ttk.Scrollbar(
            canvas_frame,
            orient=tk.HORIZONTAL
        )
        self.scrollbar_v.config(command=self.canvas.yview)
        self.scrollbar_h.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.scrollbar_v.set)
        self.canvas.config(xscrollcommand=self.scrollbar_h.set)

        self.canvas.grid(
            row=0,
            column=0,
            sticky=tk.N + tk.S + tk.E + tk.W
        )
        self.scrollbar_v.grid(row=0, column=1, sticky=tk.N + tk.S)
        self.scrollbar_h.grid(row=1, column=0, sticky=tk.E + tk.W)

        self.canvas.bind("<ButtonPress-2>", self.on_pan_button_press)
        self.canvas.bind("<B2-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_button_release)

        self.pan_start_x = None
        self.pan_start_y = None

        self.pack()

    def query_lungmap_images(self):
        lm_query_top = tk.Toplevel(bg=BACKGROUND_COLOR)
        lm_query_top.minsize(height=360, width=720)

        metadata_options_frame = tk.Frame(lm_query_top, bg=BACKGROUND_COLOR)
        metadata_options_frame.pack(
            fill=tk.X,
            expand=False,
            anchor=tk.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        metadata_options_left_frame = tk.Frame(
            metadata_options_frame,
            bg=BACKGROUND_COLOR
        )
        metadata_options_left_frame.pack(
            fill=tk.X,
            expand=False,
            anchor=tk.N,
            side=tk.LEFT,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )
        metadata_options_right_frame = tk.Frame(
            metadata_options_frame,
            bg=BACKGROUND_COLOR
        )
        metadata_options_right_frame.pack(
            fill=tk.X,
            expand=False,
            anchor=tk.N,
            side=tk.RIGHT,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        dev_stage_frame = tk.Frame(
            metadata_options_left_frame,
            bg=BACKGROUND_COLOR
        )
        dev_stage_frame.pack(
            fill=tk.X,
            expand=False,
            side=tk.TOP,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        ttk.Label(
            dev_stage_frame,
            text="Development Stage:",
            background=BACKGROUND_COLOR
        ).pack(side=tk.LEFT)
        self.dev_stage_option = ttk.Combobox(
            dev_stage_frame,
            textvariable=self.current_dev_stage,
            state='readonly'
        )
        self.dev_stage_option['values'] = sorted(DEV_STAGES)
        self.dev_stage_option.pack(side=tk.RIGHT, fill='x', expand=False)

        mag_frame = tk.Frame(
            metadata_options_left_frame,
            bg=BACKGROUND_COLOR
        )
        mag_frame.pack(
            fill=tk.X,
            expand=False,
            side=tk.TOP,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        ttk.Label(
            mag_frame,
            text="Magnification:",
            background=BACKGROUND_COLOR
        ).pack(side=tk.LEFT)
        self.mag_option = ttk.Combobox(
            mag_frame,
            textvariable=self.current_mag,
            state='readonly'
        )
        self.mag_option['values'] = MAG_VALUES
        self.mag_option.pack(side=tk.RIGHT, fill='x', expand=False)

        probe1_frame = tk.Frame(
            metadata_options_right_frame,
            bg=BACKGROUND_COLOR
        )
        probe1_frame.pack(
            fill=tk.X,
            expand=False,
            side=tk.TOP,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        ttk.Label(
            probe1_frame,
            text="Probe 1:",
            background=BACKGROUND_COLOR
        ).pack(side=tk.LEFT)
        self.probe1_option = ttk.Combobox(
            probe1_frame,
            textvariable=self.current_probe1,
            state='readonly'
        )
        self.probe1_option['values'] = sorted(PROBES)
        self.probe1_option.pack(side=tk.RIGHT, fill='x', expand=False)

        probe2_frame = tk.Frame(
            metadata_options_right_frame,
            bg=BACKGROUND_COLOR
        )
        probe2_frame.pack(
            fill=tk.X,
            expand=False,
            side=tk.TOP,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        ttk.Label(
            probe2_frame,
            text="Probe 2:",
            background=BACKGROUND_COLOR
        ).pack(side=tk.LEFT)
        self.probe2_option = ttk.Combobox(
            probe2_frame,
            textvariable=self.current_probe2,
            state='readonly'
        )
        self.probe2_option['values'] = sorted(PROBES)
        self.probe2_option.pack(side=tk.RIGHT, fill='x', expand=False)

        probe3_frame = tk.Frame(
            metadata_options_right_frame,
            bg=BACKGROUND_COLOR
        )
        probe3_frame.pack(
            fill=tk.X,
            expand=False,
            side=tk.TOP,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        ttk.Label(
            probe3_frame,
            text="Probe 3:",
            background=BACKGROUND_COLOR
        ).pack(side=tk.LEFT)
        self.probe3_option = ttk.Combobox(
            probe3_frame,
            textvariable=self.current_probe3,
            state='readonly'
        )
        self.probe3_option['values'] = sorted(PROBES)
        self.probe3_option.pack(side=tk.RIGHT, fill='x', expand=False)

        query_button_frame = tk.Frame(lm_query_top, bg=BACKGROUND_COLOR)
        query_button_frame.pack(
            fill='x',
            expand=False,
            anchor=tk.N,
            padx=PAD_SMALL,
            pady=PAD_SMALL
        )
        query_button = ttk.Button(
            query_button_frame,
            text="Run Query",
            command=self.query_images
        )
        query_button.pack(anchor=tk.E)

        file_chooser_frame = tk.Frame(lm_query_top, bg=BACKGROUND_COLOR)
        file_chooser_frame.pack(
            fill='both',
            expand=True,
            anchor=tk.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        query_results_frame = tk.Frame(
            file_chooser_frame,
            bg=BACKGROUND_COLOR,
            highlightcolor=HIGHLIGHT_COLOR,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1
        )
        query_results_scroll_bar = ttk.Scrollbar(query_results_frame, orient='vertical')
        self.query_results_list_box = tk.Listbox(
            query_results_frame,
            exportselection=False,
            height=4,
            yscrollcommand=query_results_scroll_bar.set,
            relief='flat',
            borderwidth=0,
            highlightthickness=0,
            selectbackground=HIGHLIGHT_COLOR,
            selectforeground='#ffffff'
        )
        query_results_scroll_bar.config(command=self.file_list_box.yview)
        query_results_scroll_bar.pack(side='right', fill='y')
        self.query_results_list_box.pack(fill='both', expand=True)

        query_results_frame.pack(
            fill='both',
            expand=True,
            padx=PAD_MEDIUM,
            pady=PAD_SMALL
        )

        bottom_frame = tk.Frame(lm_query_top, bg=BACKGROUND_COLOR)
        bottom_frame.pack(
            fill='x',
            expand=False,
            anchor='n',
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        progress_frame = tk.Frame(bottom_frame, bg=BACKGROUND_COLOR)
        progress_frame.pack(
            fill='both',
            expand=True,
            anchor=tk.S,
            side=tk.LEFT,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )
        self.download_progress_bar = ttk.Progressbar(progress_frame)
        self.download_progress_bar.pack(
            anchor=tk.S,
            fill='x',
            expand=True
        )

        done_button = ttk.Button(bottom_frame, text="Done", command=lm_query_top.destroy)
        done_button.pack(
            anchor=tk.E,
            side=tk.RIGHT,
            expand=False,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        download_button = ttk.Button(bottom_frame, text="Download Images", command=self.download_images)
        download_button.pack(
            anchor=tk.E,
            side=tk.RIGHT,
            expand=False,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

    def draw_polygon(self):
        self.canvas.delete("poly")
        pass

    def on_pan_button_press(self, event):
        self.canvas.config(cursor='fleur')

        # starting position for panning
        self.pan_start_x = int(self.canvas.canvasx(event.x))
        self.pan_start_y = int(self.canvas.canvasy(event.y))

    def pan_image(self, event):
        self.canvas.scan_dragto(
            event.x - self.pan_start_x,
            event.y - self.pan_start_y,
            gain=1
        )

    # noinspection PyUnusedLocal
    def on_pan_button_release(self, event):
        self.canvas.config(cursor='tcross')

    def clear_drawn_regions(self):
        self.canvas.delete("poly")

    def save_regions_json(self):
        save_file = filedialog.asksaveasfile(defaultextension=".json")
        if save_file is None:
            return

        def my_converter(o):
            if isinstance(o, np.ndarray):
                return o.tolist()

        json.dump(
            self.img_region_lut,
            save_file,
            indent=2,
            default=my_converter
        )

    # noinspection PyUnusedLocal
    def select_image(self, event):
        current_sel = self.file_list_box.curselection()
        self.current_img = self.file_list_box.get(current_sel[0])

        if self.images[self.current_img]['corr_rgb_img'] is not None:
            img_to_display = self.images[self.current_img]['corr_rgb_img']
        else:
            img_to_display = self.images[self.current_img]['rgb_img']

        image = PIL.Image.fromarray(
            img_to_display,
            'RGB'
        )
        self.tk_image = PIL.ImageTk.PhotoImage(image)
        height, width = image.size
        self.image_dims = (height, width)
        self.canvas.config(scrollregion=(0, 0, height, width))
        self.canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.tk_image
        )

    def query_images(self):
        dev_stage = self.current_dev_stage.get()
        mag = self.current_mag.get()
        probe1 = self.current_probe1.get()
        probe2 = self.current_probe2.get()
        probe3 = self.current_probe3.get()
        probes = [probe1, probe2, probe3]

        # TODO: show error dialog if any metadata field was not selected

        lm_images = lungmap_utils.client.get_images_by_metadata(
            dev_stage, mag, probes
        )

        # clear the list box & queried_images
        self.query_results_list_box.delete(0, tk.END)
        self.queried_images = {}
        for img in lm_images:
            url_parts = img['image_url']['value'].split('/')
            image_name = url_parts[-1]
            self.queried_images[image_name] = {
                'url': img['image_url']['value'],
                'dev_stage': dev_stage,
                'mag': mag,
                'probes': probes
            }

            self.query_results_list_box.insert(tk.END, image_name)

    def download_images(self):
        self.download_progress_bar.config(maximum=len(self.queried_images))
        for img_name, img_dict in sorted(self.queried_images.items()):
            image_name, tmp_img = lungmap_utils.client.get_image_from_lungmap(
                img_dict['url']
            )

            cv_img = cv2.imdecode(
                np.fromstring(
                    tmp_img,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            hsv_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

            self.images[image_name] = {
                'rgb_img': rgb_image,
                'hsv_img': hsv_image,
                'corr_rgb_img': None,
                'dev_stage': img_dict['dev_stage'],
                'mag': img_dict['mag'],
                'probes': img_dict['probes']
            }
            self.file_list_box.insert(tk.END, image_name)

            # update progress bar
            self.download_progress_bar.step()
            self.download_progress_bar.update()

    def preprocess_images(self):
        sorted_img_names = sorted(self.images.keys())

        luminance_corrected_imgs = []
        for img_name in sorted_img_names:
            lum_corr_img = micap_utils.non_uniformity_correction(
                self.images[img_name]['hsv_img']
            )
            luminance_corrected_imgs.append(lum_corr_img)

        ref_img_idx = micap_utils.find_color_correction_reference(
            luminance_corrected_imgs
        )
        self.ref_img_name = sorted_img_names[ref_img_idx]

        corr_rgb_imgs = micap_utils.color_correction(
            luminance_corrected_imgs,
            ref_img_idx
        )

        for i, img_name in enumerate(sorted_img_names):
            self.images[img_name]['corr_rgb_img'] = corr_rgb_imgs[i]

    # noinspection PyUnusedLocal
    def select_label(self, event):
        self.clear_drawn_regions()

        label = self.current_label.get()

        if label not in self.img_region_lut[self.current_img]:
            return

    # noinspection PyUnusedLocal
    def select_region(self, event):
        pass


root = themed_tk.ThemedTk()
root.set_theme('arc')
app = Application(root)
root.mainloop()

#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from skimage import io, filters, measure, morphology, util, color
from skimage.measure import find_contours
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy import ndimage as ndi
import os
import json
from skimage.segmentation import clear_border


# In[4]:


while True:
    print("\n" + "="*60)
    print("              TEM BATCH IMAGE PROCESSOR")
    print("="*60)
    print(" Pipeline Configuration Options:")
    print(" [1] Fully Automatic  (Use calculated suggestions for all images)")
    print(" [2] Manual Tweak     (Specify custom thresholds and dynamic offset scaling)")
    print(" [3] Close the program")
    print("="*60)

    while True:
        mode_selection = input("Select processing path (1, 2, or 3): ").strip()
        if mode_selection in ["1", "2", "3"]:
            break
        print("ERROR: Invalid entry! Please enter only the number 1, 2, or 3.")

    if mode_selection == "3":
        print('Are you sure you want to exit?')
        confirm = input('-> press 3 to confirm, press any other key to return to main menu: ')
        if confirm == "3":
            print("\nGood luck with the data analysis. Goodbye!")
            print("\n                   -Jason D. Orlando Ph.D.")
            break # Breaks the master loop and closes the program safely
        else:
            print("\nReturning to main menu...")
            continue  # Skips the rest of the code block and resets right back to the top options 

    # Initialize or reset the state variables for this specific run
    override_min_area = None
    override_bridging_factor = 10  # Default baseline
    offset_multiplier = 1.0       # Default baseline (100% of suggestion)

    if mode_selection == "2":
        print("\n--- CUSTOM VARIABLE MANIPULATION ---")

        # Variable 1: Minimum Particle Area
        while True:
            try:
                area_input = input("1. Enter MINIMUM PARTICLE AREA in pixels (e.g., 500): ").strip()
                override_min_area = int(area_input)
                if override_min_area <= 0:
                    print("ERROR: Area must be a positive integer greater than 0.")
                    continue
                break
            except ValueError:
                print("ERROR: Invalid number. Please enter a whole integer.")

        # Variable 2: Aggregate Bridging Factor
        while True:
            try:
                bridge_input = input("2. Enter AGGREGATE BRIDGING FACTOR (Bridges gaps to form one aggregate, default is 10 pixels): ").strip()
                override_bridging_factor = int(bridge_input)
                if override_bridging_factor <= 0:
                    print("ERROR: Bridging factor must be a positive integer greater than 0.")
                    continue
                break
            except ValueError:
                print("ERROR: Invalid number. Please enter a whole integer.")

        # Variable 3: Adaptive Offset Multiplier
        while True:
            try:
                print("\n Tip: If faint aggregates are being ignored, use a lower multiplier.")
                print("         If background noise is being captured as fake particles, use a higher multiplier.")
                mult_input = input("3. Enter ADAPTIVE OFFSET MULTIPLIER (e.g., 0.8 for stricter, 1.2 for looser, 1.0 for default): ").strip()
                offset_multiplier = float(mult_input)
                if offset_multiplier <= 0:
                    print("ERROR: Multiplier must be a positive value greater than 0.")
                    continue
                break
            except ValueError:
                print("ERROR: Invalid factor. Please enter a decimal or whole number.")

    print("\n" + "="*60)
    print(" Settings locked. Scanning folder for TEM images...")
    print("="*60 + "\n")



    folder = Path.cwd()

    default_calibration = {
        '300kx' : 0.0295, '250kx' : 0.0335, '200kx' : 0.0401, '150kx' : 0.0549,
        '120kx' : 0.0685, '100kx' : 0.0809, '80kx' : 0.0989, '60kx' : 0.1334,
        '50kx' : 0.1605, '40kx' : 0.1994,'25kx' : 0.3175, '20kx' : 0.3906,
        '15kx' : 0.5400, '12kx' : 0.6546, '10kx' : 0.7536, '8kx' : 1.0116,
        '8000x' : 1.0116, '6kx' : 1.3018, '6000x' : 1.3018, '5kx' : 1.5958,
        '5000x' : 1.5958, '4kx' : 1.9327, '4000x' : 1.9327, '3kx' : 2.6101,
        '3000x' : 2.6101, '2.5kx' : 3.2502, '2500x' : 3.2502, '2kx' : 4.0504,
        '2000x': 4.0504, '1.5kx': 5.1795, '1500x': 5.1795, '1.2kx': 6.1085,
        '1200x': 6.1085, '1kx': 7.3085, '1000x': 7.3085, '800x': 8.2870,
        '600x': 12.8533, '500x': 17.2118, '400x': 21.8337, '250x': 28.8653,
    }

    def load_calibration(file_path="calibration data/TEM calibration.txt"):
        if not os.path.exists(file_path):
            print(f" Warning: '{file_path}' not found! Using hard coded 12-May-2026 calibration")
            return default_calibration
        try:
            with open(file_path, "r") as f:
                calibration_dict = json.load(f)
            print(f"Successfully loaded calibration from '{file_path}'")
            return calibration_dict
        except json.JSONDecodeError:
            print(f" Error: '{file_path}' is formatted incorrectly! Using hard coded 12-May-2026 calibration")
            return default_calibration

    # Generate the active conversion dictionary dynamically
    pixel_nm_conversion = load_calibration()


    for active_file in folder.glob("*.jp*g"):

        # --- 3. GEOMETRY SUGGESTION ENGINE ---
        _img_raw = io.imread(active_file)
        _img_temp = color.rgb2gray(_img_raw) if _img_raw.ndim == 3 else _img_raw
        _img_temp = util.img_as_float32(_img_temp)
        img_height, img_width = _img_temp.shape

        _test_mask = _img_temp < filters.threshold_otsu(_img_temp)
        _dist_transform = ndi.distance_transform_edt(_test_mask)
        _max_thickness = np.max(_dist_transform)

        _suggested_bg_blur = int(max(img_width, img_height) * 0.10)
        _suggested_block = int(_max_thickness * 4)
        if _suggested_block % 2 == 0: _suggested_block += 1
        _suggested_block = max(21, min(_suggested_block, 151))

        _suggested_offset = round(float(np.std(_img_temp) * 0.12), 3)
        _est_particle_radius = max(3.0, _max_thickness * 0.5)
        _suggested_min_area = int((_est_particle_radius ** 2) * np.pi)
        _suggested_watershed_dist = int(_est_particle_radius * 1.5)

        print("\n" + "="*60)
        print(f"    SUGGESTED BASELINES FOR: {active_file.name}")
        print("="*60)
        print(f" -> background_blur_sigma: {_suggested_bg_blur} px")
        print(f" -> gaussian_sigma:         1")
        print(f" -> adaptive_block_size:    {_suggested_block} px")
        print(f" -> adaptive_offset:        {_suggested_offset}")
        print(f" -> min_particle_area:      {_suggested_min_area} px")
        print("="*60 + "\n")

        # =======================# 2. TUNABLE PARAMETERS====================================================
        if mode_selection == "1":         # Fully automatic assignments
            min_particle_area = int(_suggested_min_area)
            aggregate_bridging_factor = 10
            adaptive_offset = _suggested_offset
        else:
            # User defined assignments
            min_particle_area = override_min_area
            aggregate_bridging_factor = override_bridging_factor
            # Dynamically scale the unique image offset using the user multiplier
            adaptive_offset = round(float(_suggested_offset * offset_multiplier), 3)

        first_pass = 0.4
        # Image smoothing
        gaussian_sigma = 1                                                 # Higher values smooth rough surfaces and blend noise
        background_blur_sigma = _suggested_bg_blur                         # pixel diameer, Must > largest aggregate size, subtracts local background

        # determination of relative brightness for potential particles [otsu would be adaptive_block_size = whole image and adaptive_offset = 0]
        # ******auto variable generating should be very accurate for these******
        adaptive_block_size = int(_suggested_block * 1)                    # Size of local pixel neighborhood to determine relative brightness
        if adaptive_block_size % 2 == 0: adaptive_block_size += 1    

        # Data Integrity Controls
        exclude_border_particles = True # True removes chopped particles touching image borders
        scale_bar_zone_fraction = 0.03   # Fraction of the bottom screen to erase (0.05 = bottom 5%)

        # ==================================================================================================
        # 3. IMAGE PROCESSING & MEASUREMENT PIPELINE
        # ==================================================================================================
        image_raw = io.imread(active_file)
        if image_raw.ndim == 3:
            image = color.rgb2gray(image_raw)
        else:
            image = image_raw
        image = util.img_as_float32(image)
        img_height, img_width = image.shape
        label_offset = img_height * scale_bar_zone_fraction

        PDF_save_file_name = folder / f"{active_file.stem}_TEM_Aggregate_Count_Report.pdf"
        count_type = 'aggregate'
        xlsx_save_file_name = folder / f"{active_file.stem}_TEM_Aggregate_Count_Report.xlsx"
        pdf_path = PDF_save_file_name
        xlsx_path = xlsx_save_file_name
        steps = ["Loading & Filtering", "Segmenting Particles", "Calculating Metrics", "Generating Report"]

        def get_true_min_feret(region_mask):
            coords = np.argwhere(region_mask)
            if len(coords) < 3:
                return 0.0
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            min_width = float('inf')
            for angle in np.linspace(0, np.pi, 90):
                rot = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
                rotated = hull_points @ rot.T
                width = rotated[:, 0].max() - rotated[:, 0].min()
                if width < min_width:
                    min_width = width
            return min_width

        with PdfPages(pdf_path) as pdf:
            # --- STEP 1: Image Prep & Local Thresholding ---
            print(f'STEP 1 of 4: Image Prep & Local Thresholding  ({active_file.name})')

            # background flattening
            bg_map = filters.gaussian(image, sigma=background_blur_sigma)
            flattened_image = image - bg_map

            # Apply Gaussian smoothing to reduce background grain noise before thresholding
            smoothed_image = filters.gaussian(flattened_image, sigma=gaussian_sigma)

            # Adaptive threshold adapts to local brightness changes across the carbon grid
            local_thresh = filters.threshold_local(smoothed_image, block_size=adaptive_block_size, offset=adaptive_offset)
            binary_mask = smoothed_image < local_thresh

            # Baseline cleaning for visualization
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=50)
            edges = filters.sobel(smoothed_image)

            fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original')
            axes[1].imshow(edges, cmap='gray')
            axes[1].set_title('Edges (Sobel)')
            axes[2].imshow(binary_mask, cmap='gray')
            axes[2].set_title('Thresholded (Adaptive Local)')
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # --- STEP 2: Morphology Operations ---
            print(f'STEP 2 of 4: Morphology Operations  ({active_file.name})')
            binary_filled = ndi.binary_fill_holes(binary_mask)
            binary_clean_first = morphology.remove_small_objects(binary_filled, min_size=first_pass * min_particle_area)
            binary_glued = morphology.binary_closing(binary_clean_first, morphology.disk(aggregate_bridging_factor))
            binary_clean_final = morphology.remove_small_objects(binary_glued, min_size=min_particle_area)

            # Erase scale bar zone and dynamically exclude particles touching the false border
            cutoff_row = int(img_height * (1.0 - scale_bar_zone_fraction))

            if exclude_border_particles:
                # Run clear_border on only the top zone so the cutoff line acts as the bottom edge
                binary_clean_final[:cutoff_row, :] = clear_border(binary_clean_final[:cutoff_row, :])

            # Always ensure the bottom scale bar zone is completely blacked out
            binary_clean_final[cutoff_row:, :] = False


            label_image = measure.label(binary_clean_final)

            props = measure.regionprops(label_image)

            # --- STEP 3: Region Properties Loop ---
            print(f'STEP 3 of 4: Region Properties Loop  ({active_file.name})')
            results = []

            # Parse filename information safely
            try:
                sample_name = active_file.name.split("_")[0]
                mag_name = active_file.name.split('_')[1].split('(')[0].strip('x')
                mag_name = mag_name.split('.')[0]
                mag_name = mag_name + 'x'
            except IndexError:
                sample_name = "Unknown"
                mag_name = "Unknown"

            for region in props:
                y_center, x_center = region.centroid
                if y_center > (img_height * (1.0 - scale_bar_zone_fraction)):
                    continue

                region_coords = np.argwhere(region.image)
                if len(region_coords) < 3:
                    continue

                unique_y = np.unique(region_coords[:, 0])
                unique_x = np.unique(region_coords[:, 1])
                if len(unique_y) < 2 or len(unique_x) < 2:
                    continue

                max_f = region.feret_diameter_max
                min_f = get_true_min_feret(region.image)

                hull = ConvexHull(np.argwhere(region.image))
                p_convex = hull.area

                results.append({
                    'sample': sample_name,
                    'magnification': mag_name,
                    'filename': active_file.name,
                    'type': count_type,
                    'label': region.label,
                    'area': region.area,
                    'convex area': region.area_convex,
                    'convex perimeter': p_convex,
                    'max feret': max_f,
                    'min feret': min_f,
                    'circularity': (4 * np.pi * region.area_convex ) / (p_convex**2),
                    'max feret / min feret': max_f / (min_f if min_f > 0 else 1),
                    'solidity': region.area / region.area_convex
                })

            # Check if any particles were detected to avoid empty dataframe crashes
            if len(results) > 0:
                df = pd.DataFrame(results)
                df_rounded = df.round(2)
            else:
                df_rounded = pd.DataFrame(columns=['label', 'area', 'max feret', 'min feret', 'solidity'])

            # --- STEP 4: Render Plots & Table to PDF ---
            print(f'STEP 4 of 4: Rendering plots & table to PDF ({active_file.name})')

            # PAGE 2: Detection Verification
            fig2, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image, cmap='gray')
            for region in props:
                if region.area < 500:
                    continue
                y0, x0 = region.centroid
                if y0 > (img_height * (1.0 - scale_bar_zone_fraction)):
                    continue
                lbl = region.label
                d_max = region.feret_diameter_max

                ax.text(x0, y0, f"ID:{lbl}", color='yellow', fontsize=9, fontweight='bold', ha='center')
                ax.text(x0, y0 + label_offset, f"{d_max:.1f}px", color='springgreen', fontsize=8, fontweight='bold', ha='center')

                minr, minc, maxr, maxc = region.bbox
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='darkviolet', linewidth=1)
                ax.add_patch(rect)

            ax.set_title("Detection & Measurement Verification")
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

            # PAGE 3: Verification Map (With Perimeters)
            color_labels = color.label2rgb(label_image, image=image, bg_label=0)
            fig3, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(color_labels)

            contours = find_contours(binary_clean_final, level=0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='cyan', linewidth=1.5)

            for region in props:
                y0, x0 = region.centroid
                if y0 > (img_height * (1.0 - scale_bar_zone_fraction)):
                    continue
                if region.area < min_particle_area:
                    continue
                ax.text(x0, y0, str(region.label), color='white', fontsize=10, fontweight='bold', ha='center', va='center')

            ax.set_title("Verification Map (With Outlined Perimeters)")
            plt.axis('off')
            plt.tight_layout()
            pdf.savefig(fig3)
            plt.close(fig3)

            # PAGE 4: Metrics Summary Table
            if len(df_rounded) > 0:
                fig4, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('tight')
                ax.axis('off')
                table = ax.table(cellText=df_rounded.values, colLabels=df_rounded.columns, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.2)  
                ax.set_title(f"Particle Metrics Table Summary ({len(df_rounded)} Detected)", fontsize=12, pad=20)
                pdf.savefig(fig4)
                plt.close(fig4)

        print(f"Optimization Complete, image report compiled: {pdf_path.name}")
        df['max feret (nm)'] = df['max feret'] * df['magnification'].map(pixel_nm_conversion)
        df['min feret (nm)'] = df['min feret'] * df['magnification'].map(pixel_nm_conversion)
        if 'df' in locals() or 'df' in globals():
            df.to_excel(xlsx_path, index=False)
            print(f'Measurement data saved: {xlsx_path.name}')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scikit_posthocs as sp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import scipy.stats as stats
import os
import json
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[ ]:


# Setup menu 📊📈📉🧮

while True:
    print("\n" + "="*140)
    title = "📊🧮 TEM particle measurement statistics 🧮📊"
    print(f"{title:^140}")
    print("="*140)
    print(" Pipeline Configuration Options:")
    print(" [1] Run program  (Compile measurements & convert pixel to nm | Generate tables and box plots | Calculate statistical significance)")
    print(" [2] Close program")
    print("="*140)

    while True:
        mode_selection = input("Select processing path (1, or 2): ").strip()
        if mode_selection in ["1", "2"]:
            break
        print("ERROR: Invalid entry! Please enter only the number 1, 2, or 3.")

    if mode_selection == "2":
        print('Are you sure you want to exit?')
        confirm = input('-> press 2 to confirm, press any other key to return to main menu')
        if confirm == "2":
            print("\nGood luck with the data analysis. Goodbye!")
            print("\n                   -Jason D. Orlando Ph.D.")
            break # Breaks the master loop and closes the program safely
        else:
            print("\nReturning to main menu...")
            continue  # Skips the rest of the code block and resets right back to the top options

    if mode_selection == "1":
        folder = Path.cwd()
        project = str(folder).split('\\')[-1] 

        #  Compile measurement files in folder --- convert pixel to nm --- load in data for calculations 
        df_list = []
        file_list = []
        for file in folder.glob("*.xlsx"): 
            if "compiled" not in str(file).lower():
                file_df = pd.read_excel(file)
                df_list.append(file_df)
                file_name = str(file).split('\\')[-1]
                file_list.append(file_name)

        if df_list:
            df = pd.concat(df_list, axis=0, ignore_index=True)
            print
            print(f"\n{len(file_list)} files have been compiled together")
            print("-"*100)
            print(*file_list, sep='\n')
            print("-"*100 + "\n")
        else:
            print("\nNo matching Excel files were found to combine.\n")

        # Create dataframes
        request = folder.name.split(" ")[0]
        samples_list = df['sample'].unique().tolist()
        particle_dict = {}
        aggregate_dict = {}

        shape_ARs = [-np.inf, 3, np.inf]
        shape_labels = ['spherical','elongated']

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
                    print(f"Successfully loaded calibration from '{file_path}'\n")
                    return calibration_dict
                except json.JSONDecodeError:
                    print(f" Error: '{file_path}' is formatted incorrectly! Using hard coded 12-May-2026 calibration")
                    return default_calibration

        pixel_nm_conversion = load_calibration()

        df_clean = df.drop(columns=['convex area','convex perimeter','solidity','circularity','label','area'],errors='ignore')

        df_compiled_nm_particle = None
        df_compiled_nm_aggregate = None

        # create dfs for each sample particle/aggregates
        for sample in samples_list:
            df_particle = df_clean[(df_clean['sample'].astype(str).str.contains(str(sample), na=False, regex=False)) & (df_clean['type'] == 'particle')].reset_index(drop=True)
            df_particle['max feret (nm)'] = df_particle['max feret'] * df_particle['magnification'].map(pixel_nm_conversion)
            df_particle['min feret (nm)'] = df_particle['min feret'] * df_particle['magnification'].map(pixel_nm_conversion)
            df_particle['shape'] = pd.cut(df_particle['max feret / min feret'], bins=shape_ARs, labels=shape_labels, include_lowest=True)
            particle_dict[sample] = df_particle

            if df_compiled_nm_particle is not None:              # ***this creates the dataframe with nm and shape labels***
                df_compiled_nm_particle = pd.concat([df_compiled_nm_particle, df_particle], ignore_index=True)
            else:
                df_compiled_nm_particle = df_particle.copy()

        for sample in samples_list:
            df_aggregate = df_clean[(df_clean['sample'].astype(str).str.contains(str(sample), na=False, regex=False)) & (df_clean['type'] == 'aggregate')].reset_index(drop=True)
            df_aggregate['max feret (nm)'] = df_aggregate['max feret'] * df_aggregate['magnification'].map(pixel_nm_conversion)
            df_aggregate['min feret (nm)'] = df_aggregate['min feret'] * df_aggregate['magnification'].map(pixel_nm_conversion)
            aggregate_dict[sample] = df_aggregate

            if df_compiled_nm_aggregate is not None:              # ***this creates the dataframe with nm***
                df_compiled_nm_aggregate = pd.concat([df_compiled_nm_aggregate, df_aggregate], ignore_index=True)
            else:
                df_compiled_nm_aggregate = df_aggregate.copy()

        if not df_compiled_nm_particle.empty:
            df_compiled_nm_particle.to_excel(folder / f"{project} particle measurements compiled (nm).xlsx" , index=False)
            print("Compiled particle measurements (nm) have been saved as:")
            print(f"{project} particle measurements compiled (nm).xlsx\n")
        else: 
            print("NO compiled particle measurements (nm) to save\n")

        if not df_compiled_nm_aggregate.empty:
            df_compiled_nm_aggregate.to_excel(folder / f"{project} aggregate measurements compiled (nm).xlsx" , index=False)
            print("Compiled aggregate measurements (nm) have been saved as:")
            print(f"{project} aggregate measurements compiled (nm).xlsx has been saved\n")
        else: 
            print("NO compiled aggregate measurements (nm) to save\n")


        class PDFConsoleLogger:
            """Captures printed statements, displaying them to terminal and logging to PDF."""
            def __init__(self, pdf_obj):
                self.pdf = pdf_obj
                self.lines = []

            def log(self, text):
                # print(text)  # Keep normal terminal output     (***Without this it doesnt draw to terminal the stats output***)
                self.lines.extend(text.split('\n'))

            def flush_to_pdf(self, title_text):
                """Dumps all accumulated logs onto a clean text page in the PDF."""
                if not self.lines:
                    return

                max_lines_per_page = 38
                chunks = [self.lines[i:i + max_lines_per_page] for i in range(0, len(self.lines), max_lines_per_page)]

                for idx, chunk in enumerate(chunks):
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')

                    page_title = title_text if len(chunks) == 1 else f"{title_text} (Part {idx+1})"
                    ax.text(0.05, 0.95, page_title, fontsize=14, weight='bold', va='top', ha='left')

                    y_pos = 0.88
                    for line in chunk:
                        ax.text(0.05, y_pos, line, fontsize=9, fontfamily='monospace', va='top', ha='left')
                        y_pos -= 0.022

                    self.pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

                self.lines = []  # Reset logger stream

        def apply_matrix_colors(val):
            if pd.isna(val):
                return 'background-color: transparent; color: transparent;'
            elif val < 0.05:
                return 'font-weight: bold; background-color: #ffcccc; color: #990000;'
            elif val >= 0.9999:
                return 'color: #bcbcbc; font-weight: normal;'
            return 'color: #000000;'

        def get_mpl_table_styles(val):
            """Maps numerical values into hex parameters matching standard ax.table grids."""
            if pd.isna(val):
                return "#ffffff", "#ffffff", "normal"
            elif val < 0.05:
                return "#ffcccc", "#990000", "bold"
            elif val >= 0.9999:
                return "#ffffff", "#bcbcbc", "normal"
            return "#ffffff", "#000000", "normal"

        def add_matrix_table_to_pdf(pdf_obj, clean_matrix, title_text):
            """Generates a styled Matplotlib table page and appends it to the PDF."""
            cell_text = []
            cell_colors = []
            cell_text_colors = []
            cell_weights = []

            for row_idx, row in clean_matrix.iterrows():
                text_row = []
                color_row = []
                text_color_row = []
                weight_row = []
                for val in row:
                    if np.isnan(val):
                        text_row.append("")
                    else:
                        text_row.append(f"{val:.3e}")

                    bg_c, txt_c, weight = get_mpl_table_styles(val)
                    color_row.append(bg_c)
                    text_color_row.append(txt_c)
                    weight_row.append(weight)

                cell_text.append(text_row)
                cell_colors.append(color_row)
                cell_text_colors.append(text_color_row)
                cell_weights.append(weight_row)

            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.axis('off')

            table = ax.table(
                cellText=cell_text,
                cellColours=cell_colors,
                rowLabels=clean_matrix.index.tolist(),
                colLabels=clean_matrix.columns.tolist(),
                loc='center',
                cellLoc='center'
            )

            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.scale(1.2, 1.8)

            for (row_i, col_i), cell in table.get_celld().items():
                if row_i > 0 and col_i >= 0:
                    cell.get_text().set_color(cell_text_colors[row_i-1][col_i])
                    cell.get_text().set_weight(cell_weights[row_i-1][col_i])

            ax.set_title(title_text, fontsize=14, pad=20)
            pdf_obj.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ===================== Particle size statistical significance =====================
        sig_output = folder / f"{request} significant statistics.pdf"
        with PdfPages(sig_output) as pdf:
            logger = PDFConsoleLogger(pdf)

            is_normal = True
            logger.log(f"--- Normality Test Results for PARTICLE SIZE (Shapiro-Wilk) ---\n")
            for name, dataframe in particle_dict.items():
                if len(dataframe) < 3:
                    logger.log(f"Sample {name} has too few dataframe points to test normality.")
                    is_normal = False
                    continue

                stat, p_value = stats.shapiro(dataframe['max feret (nm)'])
                logger.log(f"Sample: {name} | Stat={stat:.4f}, p-value={p_value:.4e}")

                if p_value < 0.05:
                    logger.log(f"  -> Reject normality (Particle size is NOT normally distributed)\n")
                    is_normal = False
                else:
                    logger.log(f"  -> Accept normality (Particle size IS normally distributed)\n")

            if len(samples_list) == 2:
                particle_group1 = pd.to_numeric(particle_dict[samples_list[0]]['max feret (nm)'], errors='coerce').dropna().values
                particle_group2 = pd.to_numeric(particle_dict[samples_list[1]]['max feret (nm)'], errors='coerce').dropna().values

                if is_normal:
                    stat, p_val = stats.ttest_ind(particle_group1, particle_group2, equal_var=False)
                    logger.log(f"\nResult: Welch's t-test | Stat={stat:.4f}, p-value={p_val:.4e}")
                else:
                    stat, p_val = stats.mannwhitneyu(particle_group1, particle_group2, alternative='two-sided')
                    logger.log(f"\nResult: Mann-Whitney U test | Stat={stat:.4f}, p-value={p_val:.4e}")

                if p_val < 0.05:
                    logger.log("Conclusion: The two particle SIZE datasets are significantly DIFFERENT.")
                else:
                    logger.log("Conclusion: No significant difference in the two particle SIZE datasets.")

            elif len(samples_list) > 2:
                # Format molten frame needed for multi-comparison packages
                formatted_data = []
                for name, data in particle_dict.items():
                    temp_df = pd.DataFrame({'value': data['max feret (nm)'], 'group': name})
                    formatted_data.append(temp_df)
                df_posthoc = pd.concat(formatted_data, ignore_index=True)

                if is_normal:
                    stat, p_val = stats.f_oneway(*[df['max feret (nm)'].dropna().values for df in particle_dict.values()])
                    logger.log(f"\nResult: One-way ANOVA | F-Stat={stat:.4f}, p-value={p_val:.4e}")

                    logger.log("\nExecuting Tukey HSD Post-Hoc Test...")
                    tukey = pairwise_tukeyhsd(endog=df_posthoc['value'], groups=df_posthoc['group'], alpha=0.05)
                    logger.log(str(tukey))

                    # Draw visual table page using Tukey Matrix calculation
                    raw_matrix = sp.posthoc_tukey(df_posthoc, val_col='value', group_col='group')
                    table_title = "Post-Hoc Tukey Test: Particle Size (p-values)"
                else:
                    stat, p_val = stats.kruskal(*[df['max feret (nm)'].dropna().values for df in particle_dict.values()])
                    logger.log(f"\nResult: Kruskal-Wallis Test | Stat={stat:.4f}, p-value={p_val:.4e}")

                    # Draw visual table page using Dunn Matrix calculation
                    raw_matrix = sp.posthoc_dunn(df_posthoc, val_col='value', group_col='group', p_adjust='bonferroni')
                    table_title = "Post-Hoc Dunn's Test: Particle Size (Bonferroni p-values)"

                # Clean matrix shapes safely across either selection
                clean_matrix = raw_matrix.astype(float).where(~np.triu(np.ones(raw_matrix.shape), k=1).astype(bool))
                styled_matrix = clean_matrix.style.map(apply_matrix_colors).format("{:.4e}", na_rep="")
                # display(styled_matrix)

                if p_val >= 0.05:
                    logger.log("Conclusion: No significant difference in PARTICLE SIZE detected across any groups.")
            else:
                logger.log("Error: You need at least 2 datasets to run a comparison test.")

            logger.flush_to_pdf("Particle Size Statistical Report Summary")
            if len(samples_list) > 2 and 'clean_matrix' in locals():
                add_matrix_table_to_pdf(pdf, clean_matrix, table_title)
                del clean_matrix


            # ===================== Particle AR statistical significance =====================
            logger.log(f'\n' + '-'*70 + f'\n')
            is_normal = True
            logger.log(f"--- Normality Test Results for ASPECT RATIO (Shapiro-Wilk) ---\n")
            for name, dataframe in particle_dict.items():
                if len(dataframe) < 3:
                    logger.log(f"Sample {name} has too few dataframe points to test normality.")
                    is_normal = False
                    continue

                stat, p_value = stats.shapiro(dataframe['max feret / min feret'])
                logger.log(f"Sample: {name} | Stat={stat:.4f}, p-value={p_value:.4e}")

                if p_value < 0.05:
                    logger.log(f"  -> Reject normality (Aspect ratio is NOT normally distributed)\n")
                    is_normal = False
                else:
                    logger.log(f"  -> Accept normality (Aspect ratio size IS normally distributed)\n")

            if len(samples_list) == 2:
                AR_group1 = pd.to_numeric(particle_dict[samples_list[0]]['max feret / min feret'], errors='coerce').dropna().values
                AR_group2 = pd.to_numeric(particle_dict[samples_list[1]]['max feret / min feret'], errors='coerce').dropna().values

                if is_normal:
                    stat, p_val = stats.ttest_ind(AR_group1, AR_group2, equal_var=False)
                    logger.log(f"\nResult: Welch's t-test | Stat={stat:.4f}, p-value={p_val:.4e}")
                else:
                    stat, p_val = stats.mannwhitneyu(AR_group1, AR_group2, alternative='two-sided')
                    logger.log(f"\nResult: Mann-Whitney U test | Stat={stat:.4f}, p-value={p_val:.4e}")

                if p_val < 0.05:
                    logger.log("Conclusion: The two particle ASPECT RATIO datasets are significantly DIFFERENT.")
                else:
                    logger.log("Conclusion: NO significant difference in the two particle ASPECT RATIO datasets.")

            elif len(samples_list) > 2:
                formatted_data = []
                for name, data in particle_dict.items():
                    temp_df = pd.DataFrame({'value': data['max feret / min feret'], 'group': name})
                    formatted_data.append(temp_df)
                df_posthoc = pd.concat(formatted_data, ignore_index=True)

                if is_normal:
                    stat, p_val = stats.f_oneway(*[df['max feret / min feret'].dropna().values for df in particle_dict.values()])
                    logger.log(f"\nResult: One-way ANOVA | F-Stat={stat:.4f}, p-value={p_val:.4e}\n")

                    logger.log("\nExecuting Tukey HSD Post-Hoc Test...")
                    tukey = pairwise_tukeyhsd(endog=df_posthoc['value'], groups=df_posthoc['group'], alpha=0.05)
                    logger.log(str(tukey))

                    raw_matrix = sp.posthoc_tukey(df_posthoc, val_col='value', group_col='group')
                    table_title = "Post-Hoc Tukey Test: Aspect Ratio (p-values)"
                else:
                    stat, p_val = stats.kruskal(*[df['max feret / min feret'].dropna().values for df in particle_dict.values()])
                    logger.log(f"\nResult: Kruskal-Wallis Test | Stat={stat:.4f}, p-value={p_val:.4e}")

                    raw_matrix = sp.posthoc_dunn(df_posthoc, val_col='value', group_col='group', p_adjust='bonferroni')
                    table_title = "Post-Hoc Dunn's Test: Aspect Ratio (Bonferroni p-values)"

                clean_matrix = raw_matrix.astype(float).where(~np.triu(np.ones(raw_matrix.shape), k=1).astype(bool))
                styled_matrix = clean_matrix.style.map(apply_matrix_colors).format("{:.4e}", na_rep="")
                # display(styled_matrix)

                if p_val >= 0.05:
                    logger.log("Conclusion: No significant difference in ASPECT RATIO detected across any groups.")
            else:
                logger.log("Error: You need at least 2 datasets to run a comparison test.")

            logger.flush_to_pdf("Aspect Ratio Statistical Report Summary")
            if len(samples_list) > 2 and 'clean_matrix' in locals():
                add_matrix_table_to_pdf(pdf, clean_matrix, table_title)

            print("\nFiles have been saved to:")
            print(f"{folder}")
            print("-"*170)    
            print(f"Statistical significance report.................................{sig_output}")
        #======================================================================================   Particle data and statistics   ======================================================================================
        output_file_poly = f"{request} polydisperse particle size statistics.pdf"
        output_file_mono = f"{request} monodisperse particle size statistics.pdf"

        output_filepath_poly = folder / output_file_poly
        output_filepath_mono = folder / output_file_mono

        # Define the custom percentiles you want to include
        target_percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        with PdfPages(output_filepath_poly) as pdf_poly, PdfPages(output_filepath_mono) as pdf_mono:
            # --- defining all dicts for figure generation---
            aspect_dict = {}
            describe_aspect_dict = {}
            max_feret_dict = {}
            describe_max_feret_dict = {}
            min_feret_dict = {}
            describe_min_feret_dict = {}
            shape_prev_dict = {}
            max_feret_shape_dict = {}
            desc_max_feret_shape_dict = {}
            min_feret_shape_dict = {}
            desc_min_feret_shape_dict = {}
            shape_aspect_dict = {}
            desc_shape_aspect_dict = {}

            for key, value in particle_dict.items():
                aspect_dict[key] = value['max feret / min feret'].reset_index(drop=True)
                # Pass the percentiles argument here
                describe_aspect_dict[key] = value['max feret / min feret'].describe(percentiles=target_percentiles)

                # percent shapes caculated and loaded into df dict
                pct_spherical = (value['shape'] == 'spherical').mean() * 100
                pct_elongated = (value['shape'] == 'elongated').mean() * 100
                df_shape = pd.DataFrame({
                    key: [pct_spherical, pct_elongated]
                }, index=['spherical (%)', 'elongated (%)'])
                shape_prev_dict[key] = df_shape

                shape_prev = {
                    'spherical': pct_spherical, 
                    'elongated' : pct_elongated
                }
                shape_keys = list(shape_prev.keys())

                first_shape_data = value[value['shape'] == shape_keys[0]]
                second_shape_data = value[value['shape'] == shape_keys[1]]

                feret_dict_tuple = [
                    (min_feret_dict, describe_min_feret_dict, 'min feret (nm)'),
                    (max_feret_dict, describe_max_feret_dict, 'max feret (nm)')
                ]
                for boxplot_dict, table_dict, metric in feret_dict_tuple:
                    boxplot_dict[key] = value[metric].reset_index(drop=True)
                    table_dict[key] = value[metric].describe(percentiles=target_percentiles)
                    pct_below_100nm = (value[metric] < 100).mean() * 100
                    second_min_below_100 = pd.Series([pct_below_100nm], index=['< 100nm (%)'])
                    table_dict[key] = pd.concat([table_dict[key], second_min_below_100])

                feret_shape_dict = {
                     'AR<3' : first_shape_data,
                     'AR>3' : second_shape_data
                }
                shape_dict_tuple = [
                    (max_feret_shape_dict, desc_max_feret_shape_dict, 'max feret (nm)'),
                    (min_feret_shape_dict, desc_min_feret_shape_dict, 'min feret (nm)')
                ]

                for boxplot_dict, table_dict, metric in shape_dict_tuple:
                    for k4, v4 in feret_shape_dict.items():  
                        boxplot_dict[f"{key} , {k4}"] = v4[metric].reset_index(drop=True)
                        table_dict[f"{key} , {k4}"] = v4[metric].describe(percentiles=target_percentiles)
                        pct_below_100nm = (v4[metric] < 100).mean() * 100
                        below_100 = pd.Series([pct_below_100nm], index=['< 100nm (%)'])
                        table_dict[f"{key} , {k4}"] = pd.concat([table_dict[f"{key} , {k4}"], below_100])     

                        shape_aspect_dict[f"{key} , {k4}"] = v4['max feret / min feret'].reset_index(drop=True)
                        desc_shape_aspect_dict[f"{key} , {k4}"] = v4['max feret / min feret'].describe(percentiles=target_percentiles)


                for k4, v4 in feret_shape_dict.items():
                    for boxplot_dict, table_dict, metric in shape_dict_tuple:
                        boxplot_dict[f"{key} , {k4}"] = v4[metric].reset_index(drop=True)
                        table_dict[f"{key} , {k4}"] = v4[metric].describe(percentiles=target_percentiles)
                        pct_below_100nm = (v4[metric] < 100).mean() * 100
                        below_100 = pd.Series([pct_below_100nm], index=['< 100nm (%)'])
                        table_dict[f"{key} , {k4}"] = pd.concat([table_dict[f"{key} , {k4}"], below_100])     

                    shape_aspect_dict[f"{key} , {k4}"] = v4['max feret / min feret'].reset_index(drop=True)
                    desc_shape_aspect_dict[f"{key} , {k4}"] = v4['max feret / min feret'].describe(percentiles=target_percentiles)      

            # 1 Statistics Table (shape percent)----------------------------------------------------------------------------
            df_shape_desc = pd.concat(shape_prev_dict.values(), axis=1).round(2)

            # Notice: Adjust figsize height if table rows look squished
            fig, ax = plt.subplots(figsize=(8, 5)) 
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_shape_desc.values, 
                rowLabels=df_shape_desc.index, 
                colLabels=df_shape_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            plt.title(f"{request}: Shape percent distributions", pad=20)

            pdf_poly.savefig(bbox_inches="tight")
            plt.close()

            # 2 Statistics Table for AR--------------------------------------------------------------------------------------------------------
            df_aspect_desc = pd.concat(describe_aspect_dict, axis=1).round(2)

            # Notice: Adjust figsize height if table rows look squished
            fig, ax = plt.subplots(figsize=(8, 5)) 
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_aspect_desc.values, 
                rowLabels=df_aspect_desc.index, 
                colLabels=df_aspect_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            plt.title(f"{request}: Particle Aspect Ratio - Summary Statistics", pad=20)

            pdf_poly.savefig(bbox_inches="tight")
            pdf_mono.savefig(bbox_inches="tight")
            plt.close()

            # 1a Boxplot
            df_particle_aspect = pd.concat(aspect_dict, axis=1)
            df_particle_aspect.plot.box(rot=90)
            plt.title("Particle aspect ratio")
            plt.ylabel("Aspect ratio (a. u.)")
            pdf_poly.savefig(bbox_inches="tight")
            pdf_mono.savefig(bbox_inches="tight")
            plt.close()

             # 3 Statistics for AR shape contributions---------------------------------------------------------------------- 
            df_shape_aspect_dict = pd.concat(desc_shape_aspect_dict, axis=1).round(2)

            # Notice: Adjust figsize height if table rows look squished
            fig, ax = plt.subplots(figsize=(8, 5)) 
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_shape_aspect_dict.values, 
                rowLabels=df_shape_aspect_dict.index, 
                colLabels=df_shape_aspect_dict.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            plt.title("Primary and Secondary shape (Aspect Ratio) - Summary Statistics", pad=20)

            pdf_poly.savefig(bbox_inches="tight")
            plt.close()

            # 2a Boxplot
            df_shape_aspect = pd.concat(shape_aspect_dict, axis=1)
            df_shape_aspect.plot.box(rot=90)
            plt.title("Primary and Secondary shape (Aspect Ratio)")
            plt.ylabel("Aspect Ratio (a.u.)")
            pdf_poly.savefig(bbox_inches="tight")
            plt.close()


             # 4 Statistics for max feret shape contributions
            df_shape_max_desc = pd.concat(desc_max_feret_shape_dict, axis=1).round(2)

            # Notice: Adjust figsize height if table rows look squished
            fig, ax = plt.subplots(figsize=(8, 5)) 
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_shape_max_desc.values, 
                rowLabels=df_shape_max_desc.index, 
                colLabels=df_shape_max_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            plt.title("Primary and Secondary shape (max feret nm) - Summary Statistics", pad=20)

            pdf_poly.savefig(bbox_inches="tight")
            plt.close()

            # 4a Boxplot
            df_particle_max_feret = pd.concat(max_feret_shape_dict, axis=1)
            df_particle_max_feret.plot.box(rot=90)
            plt.title("Primary and Secondary shape")
            plt.ylabel("max feret (nm)")
            pdf_poly.savefig(bbox_inches="tight")
            plt.close()

            # 5 Statistics for min feret shape contributions------------------------------------------------------------------
            df_shape_min_desc = pd.concat(desc_min_feret_shape_dict, axis=1).round(2)

            # Notice: Adjust figsize height if table rows look squished
            fig, ax = plt.subplots(figsize=(8, 5)) 
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_shape_min_desc.values, 
                rowLabels=df_shape_min_desc.index, 
                colLabels=df_shape_min_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            plt.title("Primary and Secondary shape (min feret nm) - Summary Statistics", pad=20)

            pdf_poly.savefig(bbox_inches="tight")
            plt.close()

            # 5a Boxplot
            df_particle_min_feret = pd.concat(min_feret_shape_dict, axis=1)
            df_particle_min_feret.plot.box(rot=90)
            plt.title("Primary and Secondary shape")
            plt.ylabel("min feret (nm)")
            pdf_poly.savefig(bbox_inches="tight")
            plt.close()

            # 6 Total max feret size statistics table---------------------------------------------------------------------------------
            df_max_feret_desc = pd.concat(describe_max_feret_dict, axis=1).round(2)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_max_feret_desc.values, 
                rowLabels=df_max_feret_desc.index, 
                colLabels=df_max_feret_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            plt.title("Compiled particle Size (max feret nm) - Summary Statistics", pad=20)

            pdf_poly.savefig(bbox_inches="tight")
            pdf_mono.savefig(bbox_inches="tight")
            plt.close()

            # 6a. Boxplot
            df_particle_max_feret = pd.concat(max_feret_dict, axis=1)
            df_particle_max_feret.plot.box(rot=90)
            plt.title("Compiled particle Size")
            plt.ylabel("max feret (nm)")
            pdf_poly.savefig(bbox_inches="tight")
            pdf_mono.savefig(bbox_inches="tight")
            plt.close()


            # 7 Total min feret size statistics table--------------------------------------------------------------------------- 
            df_min_feret_desc = pd.concat(describe_min_feret_dict, axis=1).round(2)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_min_feret_desc.values, 
                rowLabels=df_min_feret_desc.index, 
                colLabels=df_min_feret_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            plt.title("Compiled particle Size (min feret nm) - Summary Statistics", pad=20)

            pdf_poly.savefig(bbox_inches="tight")
            pdf_mono.savefig(bbox_inches="tight")
            plt.close()

            # 7a Boxplot
            df_particle_min_feret = pd.concat(min_feret_dict, axis=1)
            df_particle_min_feret.plot.box(rot=90)
            plt.title("Compiled particle Size")
            plt.ylabel("min feret (nm)")
            pdf_poly.savefig(bbox_inches="tight")
            pdf_mono.savefig(bbox_inches="tight")
            plt.close()

            print(f"Monodisperse particle statistics report created.................{output_file_mono}") 
            print(f"Polydisperse particle statistics report created.................{output_file_poly}")

        #======================================================================================   Aggregate data and statistics   ======================================================================================
        output_file_aggregate = f"{request} aggregate size statistics.pdf"
        output_filepath_aggregate = folder / output_file_aggregate

        # Define the custom percentiles you want to include
        target_percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        with PdfPages(output_filepath_aggregate) as pdf:
            # --- SECTION 1: ASPECT RATIO ---
            aspect_dict = {}
            describe_aspect_dict = {}

            for key, value in aggregate_dict.items():
                value['max feret / min feret'] = pd.to_numeric(value['max feret / min feret'], errors='coerce')
                aspect_dict[key] = value['max feret / min feret'].reset_index(drop=True)
                # Pass the percentiles argument here
                describe_aspect_dict[key] = value['max feret / min feret'].describe(percentiles=target_percentiles)

            # 1a. Boxplot
            df_aggregate_aspect = pd.concat(aspect_dict, axis=1)
            df_aggregate_aspect.plot.box(rot=90)
            plt.title(f"{request}: Aggregate aspect ratio")
            plt.ylabel("Aspect ratio (a. u.)")
            pdf.savefig(bbox_inches="tight")
            plt.close()



            # 1b. Statistics Table
            df_aspect_desc = pd.concat(describe_aspect_dict, axis=1).round(2)

            # Notice: Adjust figsize height if table rows look squished
            fig, ax = plt.subplots(figsize=(8, 5)) 
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_aspect_desc.values, 
                rowLabels=df_aspect_desc.index, 
                colLabels=df_aspect_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            plt.title("Aggregate Aspect Ratio - Summary Statistics", pad=20)

            pdf.savefig(bbox_inches="tight")
            plt.close()


            # --- SECTION 2: MAX FERET SIZE ---
            max_feret_dict = {}
            describe_max_feret_dict = {}

            for key, value in aggregate_dict.items():
                value['max feret (nm)'] = pd.to_numeric(value['max feret (nm)'], errors='coerce')
                max_feret_dict[key] = value['max feret (nm)'].reset_index(drop=True)
                # Pass the percentiles argument here as well
                describe_max_feret_dict[key] = value['max feret (nm)'].describe(percentiles=target_percentiles)

            # 2a. Boxplot
            df_aggregate_max_feret = pd.concat(max_feret_dict, axis=1)
            df_aggregate_max_feret.plot.box(rot=90)
            plt.title("Aggregate size")
            plt.ylabel("max feret (nm)")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            # 2b. Statistics Table
            df_max_feret_desc = pd.concat(describe_max_feret_dict, axis=1).round(2)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_max_feret_desc.values, 
                rowLabels=df_max_feret_desc.index, 
                colLabels=df_max_feret_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            plt.title("Aggregate Size (max feret nm) - Summary Statistics", pad=20)

            pdf.savefig(bbox_inches="tight")
            plt.close()

            # --- SECTION 3: MIN FERET SIZE ---
            min_feret_dict = {}
            describe_min_feret_dict = {}

            for key, value in aggregate_dict.items():
                value['min feret (nm)'] = pd.to_numeric(value['min feret (nm)'], errors='coerce')
                min_feret_dict[key] = value['min feret (nm)'].reset_index(drop=True)
                # Pass the percentiles argument here as well
                describe_min_feret_dict[key] = value['min feret (nm)'].describe(percentiles=target_percentiles)

            # 2a. Boxplot
            df_aggregate_max_feret = pd.concat(min_feret_dict, axis=1)
            df_aggregate_max_feret.plot.box(rot=90)
            plt.title("Aggregate size")
            plt.ylabel("min feret (nm)")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            # 2b. Statistics Table
            df_max_feret_desc = pd.concat(describe_min_feret_dict, axis=1).round(2)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=df_max_feret_desc.values, 
                rowLabels=df_max_feret_desc.index, 
                colLabels=df_max_feret_desc.columns, 
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            plt.title("Aggregate Size (min feret nm) - Summary Statistics", pad=20)

            pdf.savefig(bbox_inches="tight")
            plt.close()

            print(f"Aggregate statistics report created.............................{output_file_aggregate}")


# In[ ]:





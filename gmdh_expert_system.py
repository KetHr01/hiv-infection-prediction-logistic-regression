import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ── Кольорова палітра світлої теми ──
BG_MAIN    = "#f1f5f9"
BG_CARD    = "#ffffff"
BG_DARK    = "#e2e8f0"
BORDER     = "#cbd5e1"
TEXT_PRI   = "#0f172a"
TEXT_SEC   = "#475569"
TEXT_MUTED = "#94a3b8"
BLUE       = "#2563eb"
GREEN      = "#16a34a"
RED        = "#dc2626"
YELLOW     = "#ca8a04"
PURPLE     = "#7c3aed"
PLOT_BG    = "#f8fafc"


class GMDH_Final_Expert_System(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Аналітична система діагностики (KPI BS-23)")
        self.geometry("1450x900")
        self.configure(fg_color=BG_MAIN)

        self.data = {
            "Випадкове розбиття": {
                "f1": 0.7103, "pr": 0.6909, "re": 0.7308, "feats": 9,
                "cm": [145, 17, 14, 38],
                "val_f1": 0.7619, "val_feats": 9, "val_fin": 3.84, "val_fout": 2.71,
                "model": "z = -1.6630 - 1.8428*time - 0.9515*offtrt - 0.5652*log_cd420 - 0.2781*race - 0.3108*cd4_ratio_20 + 0.2488*age_x_cd40 + 0.1787*severe_immunodef + 0.1783*preanti - 0.1641*karnof"
            },
            "Метрика Махаланобіса": {
                "f1": 0.6792, "pr": 0.6667, "re": 0.6923, "feats": 11,
                "cm": [143, 18, 16, 36],
                "val_f1": 0.7115, "val_feats": 11, "val_fin": 3.2, "val_fout": 2.5,
                "model": "z = -1.7947 - 0.9887*time - 1.0816*offtrt - 0.5370*log_cd420 - 0.4483*cd4_ratio_20 - 0.3086*race + 0.2878*age_x_cd40 + 0.1711*symptom - 0.9673*time_x_karnof + 0.1855*z30 - 0.1885*drugs - 0.1664*hemo"
            },
            "Крос-валідація": {
                "f1": 0.7642, "pr": 0.7570, "re": 0.7714, "feats": 15,
                "cm": [297, 26, 24, 81],
                "val_f1": 0.7642, "val_feats": 15, "val_fin": None, "val_fout": None,
                "model": "z = -1.6868 - 1.7874*time + 0.1688*age + 0.2425*z30 + 0.2689*preanti - 0.2191*race + 0.1166*gender - 0.1988*str2 + 0.0291*strat - 0.8558*offtrt - 0.0787*cd420 - 0.2170*cd4_ratio_0 - 0.0842*cd4_change + 0.0667*age_x_cd40 - 0.5259*log_cd420"
            }
        }

        self.cv_folds = {
            "Фолд 1": {"f1": 0.6878, "pr": 0.6496, "re": 0.7308, "acc": 0.8388, "model": "z = -1.7203 - 1.9069*time - 0.0924*trt - 0.1953*drugs + 0.3263*z30 - 0.3254*race - 0.9162*offtrt + 0.2568*cd4_ratio_0 - 0.7402*cd4_ratio_20 - 0.1353*cd8_change + 0.2609*severe_immunodef + 0.1771*cd4_dropped"},
            "Фолд 2": {"f1": 0.6927, "pr": 0.7030, "re": 0.6827, "acc": 0.8528, "model": "z = -1.7720 - 1.8322*time - 0.0905*trt + 0.2466*age + 0.1950*z30 + 0.2131*preanti - 0.2866*race - 0.1226*strat - 0.9896*offtrt - 0.0607*cd40 - 0.7785*cd420 + 0.1710*cd40_sq"},
            "Фолд 3": {"f1": 0.7155, "pr": 0.6484, "re": 0.7981, "acc": 0.8458, "model": "z = -1.6669 - 1.7669*time + 0.1691*age + 0.5421*preanti + 0.1517*gender - 0.8408*offtrt + 0.1379*cd40 - 0.7740*cd420 + 0.1907*cd820 - 0.1943*cd4_pct_change"},
            "Фолд 4": {"f1": 0.7642, "pr": 0.7570, "re": 0.7714, "acc": 0.8832, "model": "z = -1.6868 - 1.7874*time + 0.1688*age + 0.2425*z30 + 0.2689*preanti - 0.2191*race - 0.8558*offtrt - 0.0787*cd420 - 0.2170*cd4_ratio_0 - 0.5259*log_cd420"},
            "Фолд 5": {"f1": 0.7411, "pr": 0.6917, "re": 0.7981, "acc": 0.8642, "model": "z = -1.6873 - 0.8948*time + 0.1514*age - 0.6318*drugs + 0.2293*z30 - 0.8946*offtrt - 0.7884*cd420 - 0.4222*cd4_ratio_20 - 1.0028*time_x_karnof + 0.2151*cd40_sq"}
        }

        self.setup_ui()

    # ════════════════════════════════════════════
    def setup_ui(self):
        self.tabview = ctk.CTkTabview(
            self, fg_color=BG_MAIN,
            segmented_button_selected_color=BLUE,
            segmented_button_selected_hover_color="#1d4ed8",
            segmented_button_unselected_color=BG_DARK,
            segmented_button_unselected_hover_color=BORDER,
            text_color=TEXT_PRI)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        self.tab_dash    = self.tabview.add("🔍 Окремі моделі")
        self.tab_comp    = self.tabview.add("📊 Порівняльний аналіз")
        self.tab_predict = self.tabview.add("🧮 Розрахунок результату")

        self.canvas_dash = None
        self.setup_dashboard_tab()
        self.setup_comparison_tab()
        self.setup_prediction_tab()

        self.method_var.set("Крос-валідація")
        self.update_dashboard()

    # ════════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════════
    def setup_dashboard_tab(self):
        top_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        top_frame.pack(pady=15, fill="x", padx=50)

        self.method_var = ctk.StringVar()
        self.menu = ctk.CTkOptionMenu(
            top_frame, values=list(self.data.keys()),
            variable=self.method_var, command=lambda x: self.update_dashboard(),
            fg_color=BLUE, button_color="#1d4ed8", text_color="white")
        self.menu.pack(side="left", padx=10)

        self.cv_btn = ctk.CTkButton(
            top_frame, text="Деталі 5-ти фолдів CV",
            fg_color=PURPLE, hover_color="#6d28d9", text_color="white",
            command=self.open_cv_comparison)

        metrics_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        metrics_frame.pack(fill="x", padx=50)
        self.f1_card = self._metric_card(metrics_frame, "F1-SCORE",  BLUE)
        self.pr_card = self._metric_card(metrics_frame, "PRECISION", GREEN)
        self.re_card = self._metric_card(metrics_frame, "RECALL",    RED)

        vis_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        vis_frame.pack(fill="both", expand=True, padx=50, pady=20)

        left_side = ctk.CTkFrame(vis_frame, fg_color="transparent")
        left_side.pack(side="left", fill="both", expand=True)

        cm_box = ctk.CTkFrame(left_side, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        cm_box.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(cm_box, text="Матриця помилок", font=("Arial", 14, "bold"),
                     text_color=TEXT_PRI).pack(pady=5)
        grid = ctk.CTkFrame(cm_box, fg_color="transparent")
        grid.pack(pady=10)
        self.cm_cells = []
        for i in range(4):
            cell = ctk.CTkFrame(grid, fg_color=BG_DARK, width=110, height=70)
            cell.grid(row=i//2, column=i%2, padx=5, pady=5)
            lbl = ctk.CTkLabel(cell, text="0", font=("Arial", 18, "bold"), text_color=TEXT_PRI)
            lbl.place(relx=0.5, rely=0.5, anchor="center")
            self.cm_cells.append(lbl)

        model_box = ctk.CTkFrame(left_side, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        model_box.pack(fill="both", expand=True)
        ctk.CTkLabel(model_box, text="Рівняння регресії (GMDH)", font=("Arial", 14, "bold"),
                     text_color=TEXT_PRI).pack(pady=5)
        self.model_text = ctk.CTkTextbox(model_box, font=("Consolas", 12),
                                         fg_color=BG_DARK, text_color=TEXT_PRI)
        self.model_text.pack(fill="both", expand=True, padx=10, pady=10)

        self.plot_frame_dash = ctk.CTkFrame(vis_frame, fg_color=BG_CARD,
                                             border_width=1, border_color=BORDER)
        self.plot_frame_dash.pack(side="right", fill="both", expand=True, padx=(20, 0))

    def _metric_card(self, parent, title, color):
        f = ctk.CTkFrame(parent, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        f.pack(side="left", expand=True, fill="x", padx=10)
        ctk.CTkLabel(f, text=title, font=("Arial", 11), text_color=TEXT_SEC).pack(pady=(5, 0))
        lbl = ctk.CTkLabel(f, text="0.0", font=("Arial", 28, "bold"), text_color=color)
        lbl.pack(pady=10)
        return lbl

    def update_dashboard(self):
        m = self.method_var.get()
        d = self.data[m]

        if m == "Крос-валідація":
            self.cv_btn.pack(side="left", padx=10)
        else:
            self.cv_btn.pack_forget()

        self.f1_card.configure(text=f"{d['f1']:.4f}")
        self.pr_card.configure(text=f"{d['pr']:.4f}")
        self.re_card.configure(text=f"{d['re']:.4f}")
        for i, val in enumerate(d['cm']):
            self.cm_cells[i].configure(text=str(val))
        self.model_text.delete("1.0", "end")
        self.model_text.insert("1.0", d['model'])

        if self.canvas_dash:
            self.canvas_dash.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(5, 4), facecolor=PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        bar_colors = [BLUE if k == m else BG_DARK for k in self.data.keys()]
        ax.bar(list(self.data.keys()), [v['f1'] for v in self.data.values()],
               color=bar_colors, edgecolor=BORDER)
        ax.tick_params(colors=TEXT_PRI, labelsize=8)
        ax.spines[:].set_color(BORDER)
        ax.set_ylim(0, 1.0)
        fig.patch.set_facecolor(PLOT_BG)

        self.canvas_dash = FigureCanvasTkAgg(fig, master=self.plot_frame_dash)
        self.canvas_dash.draw()
        self.canvas_dash.get_tk_widget().pack(fill="both", expand=True)

    # ════════════════════════════════════════════
    # TAB 2
    # ════════════════════════════════════════════
    def setup_comparison_tab(self):
        scroll = ctk.CTkScrollableFrame(self.tab_comp, fg_color=BG_MAIN)
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        graph_container = ctk.CTkFrame(scroll, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        graph_container.pack(fill="x", padx=10, pady=10)

        short_labels = ["Вип. розб.", "Махаланобіс", "Крос-вал."]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=PLOT_BG)
        plt.subplots_adjust(bottom=0.22, wspace=0.4)

        plots = [
            (axes[0], [d['f1']     for d in self.data.values()], "F1-Score (Тест)",      "#2563eb"),
            (axes[1], [d['pr']     for d in self.data.values()], "Precision (Тест)",     "#16a34a"),
            (axes[2], [d['re']     for d in self.data.values()], "Recall (Тест)",        "#dc2626"),
            (axes[3], [d['val_f1'] for d in self.data.values()], "F1-Score (Валідація)", "#ca8a04"),
        ]
        for ax, vals, title, color in plots:
            ax.set_facecolor(PLOT_BG)
            bars = ax.bar(short_labels, vals, color=color, alpha=0.85, edgecolor=BORDER)
            ax.set_title(title, color=TEXT_PRI, fontsize=9, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='x', colors=TEXT_PRI, rotation=15, labelsize=7)
            ax.tick_params(axis='y', colors=TEXT_PRI, labelsize=8)
            ax.spines[:].set_color(BORDER)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.4f}", ha='center', va='bottom', color=TEXT_PRI, fontsize=7)

        fig.patch.set_facecolor(PLOT_BG)
        canvas_comp = FigureCanvasTkAgg(fig, master=graph_container)
        canvas_comp.draw()
        canvas_comp.get_tk_widget().pack(fill="x")

        self._section_label(scroll, "📋 Результати на екзаменаційній (тестовій) вибірці")
        table_frame = ctk.CTkFrame(scroll, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        table_frame.pack(fill="x", padx=10, pady=(0, 10))

        headers_test   = ["Метод розбиття", "F1", "Precision", "Recall", "Ознак"]
        col_widths_test = [200, 90, 90, 90, 70]
        for j, (h, w) in enumerate(zip(headers_test, col_widths_test)):
            ctk.CTkLabel(table_frame, text=h, font=("Arial", 12, "bold"),
                         text_color=TEXT_PRI, width=w, anchor="center").grid(row=0, column=j, padx=8, pady=10)
        for i, (name, d) in enumerate(self.data.items()):
            row_data   = [name, f"{d['f1']:.4f}", f"{d['pr']:.4f}", f"{d['re']:.4f}", str(d['feats'])]
            row_colors = [TEXT_PRI, BLUE, GREEN, RED, YELLOW]
            for j, (val, col, w) in enumerate(zip(row_data, row_colors, col_widths_test)):
                ctk.CTkLabel(table_frame, text=val, text_color=col, width=w,
                             anchor="center").grid(row=i+1, column=j, padx=8, pady=6)

        self._section_label(scroll, "📋 Результати на валідаційній вибірці")
        table_val = ctk.CTkFrame(scroll, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        table_val.pack(fill="x", padx=10, pady=(0, 20))

        headers_val   = ["Метод розбиття", "Val F1 (найкраще)", "Ознак (вал.)", "F_in", "F_out"]
        col_widths_val = [200, 160, 120, 80, 80]
        for j, (h, w) in enumerate(zip(headers_val, col_widths_val)):
            ctk.CTkLabel(table_val, text=h, font=("Arial", 12, "bold"),
                         text_color=TEXT_PRI, width=w, anchor="center").grid(row=0, column=j, padx=8, pady=10)

        val_rows = [
            ("Випадкове розбиття", 0.7619, 9,  3.84, 2.71),
            ("Метрика Махаланобіса", 0.7115, 11, 3.2,  2.5),
            ("Крос-валідація",     0.7642, 15, "—",  "—"),
        ]
        for i, row in enumerate(val_rows):
            name, vf1, vfeats, fin, fout = row
            row_data   = [name, f"{vf1:.4f}", str(vfeats),
                          str(fin) if fin != "—" else "—",
                          str(fout) if fout != "—" else "—"]
            row_colors = [TEXT_PRI, YELLOW, YELLOW, PURPLE, PURPLE]
            for j, (val, col, w) in enumerate(zip(row_data, row_colors, col_widths_val)):
                ctk.CTkLabel(table_val, text=val, text_color=col, width=w,
                             anchor="center").grid(row=i+1, column=j, padx=8, pady=6)

    def _section_label(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=("Arial", 13, "bold"),
                     text_color=TEXT_SEC).pack(anchor="w", padx=15, pady=(15, 4))

    # ════════════════════════════════════════════
    # TAB 3
    # ════════════════════════════════════════════
    def setup_prediction_tab(self):
        main = ctk.CTkScrollableFrame(self.tab_predict, fg_color=BG_MAIN)
        main.pack(fill="both", expand=True, padx=20, pady=10)

        desc = ctk.CTkFrame(main, fg_color="#dbeafe", border_width=1, border_color="#93c5fd")
        desc.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(desc, text="📌 Введення даних пацієнта для прогнозу",
                     font=("Arial", 14, "bold"), text_color=BLUE).pack(anchor="w", padx=15, pady=(10, 2))
        ctk.CTkLabel(desc,
                     text="Заповніть поля нижче значеннями з набору даних. Усі поля є обов'язковими.\n"
                          "Після натискання «Розрахувати» система виведе прогноз за трьома моделями GMDH.",
                     font=("Arial", 11), text_color=TEXT_SEC,
                     justify="left", wraplength=900).pack(anchor="w", padx=15, pady=(0, 10))

        self.field_specs = [
            ("time",    "time",    "int",   0,   9999, "Час спостереження (днів), ціле число ≥ 0"),
            ("trt",     "trt",     "int",   1,   4,    "Група лікування: 1, 2, 3 або 4"),
            ("age",     "age",     "float", 18,  80,   "Вік пацієнта (років), 18–80"),
            ("wtkg",    "wtkg",    "float", 30,  150,  "Маса тіла (кг), 30–150"),
            ("hemo",    "hemo",    "int",   0,   1,    "Гемофілія: 0 = ні, 1 = так"),
            ("homo",    "homo",    "int",   0,   1,    "Гомосексуальні контакти: 0 = ні, 1 = так"),
            ("drugs",   "drugs",   "int",   0,   1,    "Вживання наркотиків: 0 = ні, 1 = так"),
            ("karnof",  "karnof",  "int",   50,  100,  "Шкала Карнофського (50–100, крок 10)"),
            ("oprior",  "oprior",  "int",   0,   1,    "Попередні опортуністичні інфекції: 0 = ні, 1 = так"),
            ("z30",     "z30",     "int",   0,   1,    "CD4 < 300 за 30 днів до включення: 0 = ні, 1 = так"),
            ("preanti", "preanti", "float", 0,   9999, "Дні попередньої антиретровірусної терапії, ≥ 0"),
            ("race",    "race",    "int",   0,   1,    "Раса: 0 = білий, 1 = не-білий"),
            ("gender",  "gender",  "int",   0,   1,    "Стать: 0 = жінка, 1 = чоловік"),
            ("str2",    "str2",    "int",   0,   1,    "Страта CD4: 0 = ≥50, 1 = <50"),
            ("strat",   "strat",   "int",   1,   3,    "Страта: 1, 2 або 3"),
            ("symptom", "symptom", "int",   0,   1,    "Наявність симптомів: 0 = ні, 1 = так"),
            ("treat",   "treat",   "int",   0,   1,    "Ознака лікування: 0 = ні, 1 = так"),
            ("offtrt",  "offtrt",  "int",   0,   1,    "Зійшов з лікування: 0 = ні, 1 = так"),
            ("cd40",    "cd40",    "float", 0,   2000, "CD4 клітини на початку (клітин/мм³), ≥ 0"),
            ("cd420",   "cd420",   "float", 0,   2000, "CD4 клітини через 20 тижнів, ≥ 0"),
            ("cd80",    "cd80",    "float", 0,   5000, "CD8 клітини на початку (клітин/мм³), ≥ 0"),
            ("cd820",   "cd820",   "float", 0,   5000, "CD8 клітини через 20 тижнів, ≥ 0"),
        ]

        inp_frame = ctk.CTkFrame(main, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        inp_frame.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(inp_frame, text="Вхідні дані", font=("Arial", 13, "bold"),
                     text_color=TEXT_PRI).pack(anchor="w", padx=15, pady=(10, 5))

        grid_f = ctk.CTkFrame(inp_frame, fg_color="transparent")
        grid_f.pack(fill="x", padx=15, pady=(0, 10))

        self.input_vars = {}
        for idx, (key, label, typ, mn, mx, desc) in enumerate(self.field_specs):
            row, col = divmod(idx, 3)
            cell = ctk.CTkFrame(grid_f, fg_color=BG_DARK, border_width=1, border_color=BORDER)
            cell.grid(row=row, column=col, padx=6, pady=5, sticky="ew")
            grid_f.columnconfigure(col, weight=1)
            ctk.CTkLabel(cell, text=label, font=("Arial", 11, "bold"),
                         text_color=BLUE).pack(anchor="w", padx=8, pady=(6, 0))
            ctk.CTkLabel(cell, text=desc, font=("Arial", 9),
                         text_color=TEXT_MUTED, wraplength=280, justify="left").pack(anchor="w", padx=8)
            var = ctk.StringVar()
            ctk.CTkEntry(cell, textvariable=var, placeholder_text=f"Введіть {label}",
                         font=("Arial", 11), fg_color=BG_CARD,
                         text_color=TEXT_PRI, border_color=BORDER).pack(fill="x", padx=8, pady=(2, 8))
            self.input_vars[key] = var

        ex_frame = ctk.CTkFrame(main, fg_color="transparent")
        ex_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(ex_frame, text="Приклади з датасету:", font=("Arial", 11),
                     text_color=TEXT_SEC).pack(side="left", padx=(0, 10))
        ctk.CTkButton(ex_frame, text="Приклад 1 (незаражений)",
                      fg_color="#dcfce7", hover_color="#bbf7d0", text_color=GREEN,
                      border_width=1, border_color=GREEN, width=200,
                      command=lambda: self._fill_example(1)).pack(side="left", padx=5)
        ctk.CTkButton(ex_frame, text="Приклад 2 (заражений)",
                      fg_color="#fee2e2", hover_color="#fecaca", text_color=RED,
                      border_width=1, border_color=RED, width=200,
                      command=lambda: self._fill_example(2)).pack(side="left", padx=5)
        ctk.CTkButton(ex_frame, text="Очистити",
                      fg_color=BG_DARK, hover_color=BORDER, text_color=TEXT_SEC,
                      border_width=1, border_color=BORDER, width=120,
                      command=self._clear_inputs).pack(side="left", padx=5)

        ctk.CTkButton(main, text="▶  Розрахувати прогноз за трьома моделями",
                      font=("Arial", 13, "bold"), height=45,
                      fg_color=BLUE, hover_color="#1d4ed8", text_color="white",
                      command=self._calculate).pack(pady=15)

        self.result_frame = ctk.CTkFrame(main, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        self.result_frame.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(self.result_frame, text="Результати прогнозу",
                     font=("Arial", 13, "bold"), text_color=TEXT_PRI).pack(anchor="w", padx=15, pady=(10, 5))

        res_row = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        res_row.pack(fill="x", padx=15, pady=(0, 15))

        self.result_cards = []
        for name in ["Випадкове розбиття", "Метрика Махаланобіса", "Крос-валідація (Фолд 4)"]:
            card = ctk.CTkFrame(res_row, border_width=2, border_color=BORDER, fg_color=BG_DARK)
            card.pack(side="left", expand=True, fill="x", padx=8)
            ctk.CTkLabel(card, text=name, font=("Arial", 11, "bold"),
                         text_color=TEXT_SEC, wraplength=200).pack(pady=(10, 4))
            v_lbl = ctk.CTkLabel(card, text="—", font=("Arial", 22, "bold"), text_color=TEXT_PRI)
            v_lbl.pack()
            p_lbl = ctk.CTkLabel(card, text="P = —", font=("Arial", 12), text_color=TEXT_SEC)
            p_lbl.pack(pady=(2, 4))
            z_lbl = ctk.CTkLabel(card, text="z = —", font=("Arial", 10), text_color=TEXT_MUTED)
            z_lbl.pack(pady=(0, 10))
            self.result_cards.append((v_lbl, p_lbl, z_lbl, card))

    def _fill_example(self, n):
        examples = {
            1: dict(time=948, trt=2, age=48, wtkg=89.8128, hemo=0, homo=0, drugs=0,
                    karnof=100, oprior=0, z30=0, preanti=0, race=0, gender=0, str2=0,
                    strat=1, symptom=0, treat=1, offtrt=0, cd40=422, cd420=477,
                    cd80=566, cd820=324),
            2: dict(time=1002, trt=3, age=61, wtkg=49.4424, hemo=0, homo=0, drugs=0,
                    karnof=90, oprior=0, z30=1, preanti=895, race=0, gender=0, str2=1,
                    strat=3, symptom=0, treat=1, offtrt=0, cd40=162, cd420=150,
                    cd80=392, cd820=678),
        }
        for key, var in self.input_vars.items():
            var.set(str(examples[n].get(key, "")))

    def _clear_inputs(self):
        for var in self.input_vars.values():
            var.set("")

    def _get_val(self, key, typ):
        raw = self.input_vars[key].get().strip()
        if raw == "":
            raise ValueError(f"Поле '{key}' порожнє")
        return int(float(raw)) if typ == "int" else float(raw)

    def _calculate(self):
        try:
            v = {}
            for key, label, typ, mn, mx, desc in self.field_specs:
                v[key] = self._get_val(key, typ)
        except ValueError as e:
            self._show_error(str(e))
            return

        # ── Точні параметри StandardScaler (всі ознаки, включно з бінарними) ──
        MEAN = {
            "time": 879.0981767180925, "age": 35.24824684431977,
            "karnof": 95.44647031323048, "preanti": 379.17578307620386,
            "cd40": 350.5011687704535, "cd420": 371.3071528751753,
            "cd80": 986.6273959794296, "cd820": 935.369798971482,
            "cd4_ratio_0": 0.42663392518297516, "cd4_ratio_20": 0.4631229185992576,
            "cd4_change": 20.805984104721833, "age_x_cd40": 12312.952781673679,
            "time_x_karnof": 84069.58391771856, "log_cd420": 5.839926380441065,
            "severe_immunodef": 0.0766713417484806,
            "offtrt": 0.3627863487611033, "race": 0.2884525479195886,
            "symptom": 0.17297802711547453, "z30": 0.5502571294997662,
            "drugs": 0.131369798971482, "hemo": 0.08415147265077139,
            "gender": 0.8279569892473119, "str2": 0.5857877512856475,
            "strat": 1.9798971482000935, "trt_3": 0.26227208976157085,
        }
        STD = {
            "time": 292.20599548523495, "age": 8.70699022551006,
            "karnof": 5.899605015695356, "preanti": 468.54796236535805,
            "cd40": 118.54614215589578, "cd420": 144.6010959574959,
            "cd80": 480.0854889366221, "cd820": 444.8720238149072,
            "cd4_ratio_0": 0.3062526044001905, "cd4_ratio_20": 0.2450444012114444,
            "cd4_change": 122.29219617442034, "age_x_cd40": 5146.475211036888,
            "time_x_karnof": 28680.95920650138, "log_cd420": 0.41510440847841007,
            "severe_immunodef": 0.26606925245689006,
            "offtrt": 0.4808039246030449, "race": 0.4530426861988681,
            "symptom": 0.37822827664085706, "z30": 0.49746780894289405,
            "drugs": 0.3378043440953275, "hemo": 0.27761484524693175,
            "gender": 0.37741782311363964, "str2": 0.49258548672220526,
            "strat": 0.898842322645251, "trt_3": 0.43986979970631007,
        }

        def s(feat, val):
            return (val - MEAN[feat]) / STD[feat]

        cd40 = v["cd40"]; cd420 = v["cd420"]
        cd80 = v["cd80"]; cd820 = v["cd820"]
        time = v["time"]; age   = v["age"]
        karnof = v["karnof"]; preanti = v["preanti"]

        cd4_ratio_0      = cd40  / (cd80  + 1e-5)
        cd4_ratio_20     = cd420 / (cd820 + 1e-5)
        cd4_change       = cd420 - cd40
        age_x_cd40       = age   * cd40
        time_x_karnof    = time  * karnof
        log_cd420        = math.log1p(cd420)
        severe_immunodef = 1 if cd40 < 200 else 0
        trt_3            = 1 if v["trt"] == 3 else 0

        z1 = (-1.6630
              - 1.8428 * s("time",             time)
              - 0.9515 * s("offtrt",           v["offtrt"])
              - 0.5652 * s("log_cd420",        log_cd420)
              - 0.2781 * s("race",             v["race"])
              - 0.3108 * s("cd4_ratio_20",     cd4_ratio_20)
              + 0.2488 * s("age_x_cd40",       age_x_cd40)
              + 0.1787 * s("severe_immunodef", severe_immunodef)
              + 0.1783 * s("preanti",          preanti)
              - 0.1641 * s("karnof",           karnof))

        z2 = (-1.7947
              - 0.9887 * s("time",          time)
              - 1.0816 * s("offtrt",        v["offtrt"])
              - 0.5370 * s("log_cd420",     log_cd420)
              - 0.4483 * s("cd4_ratio_20",  cd4_ratio_20)
              - 0.3086 * s("race",          v["race"])
              + 0.2878 * s("age_x_cd40",    age_x_cd40)
              + 0.1711 * s("symptom",       v["symptom"])
              - 0.9673 * s("time_x_karnof", time_x_karnof)
              + 0.1855 * s("z30",           v["z30"])
              - 0.1885 * s("drugs",         v["drugs"])
              - 0.1664 * s("hemo",          v["hemo"]))

        z3 = (-1.6868
              - 1.7874 * s("time",        time)
              + 0.1688 * s("age",         age)
              + 0.2425 * s("z30",         v["z30"])
              + 0.2689 * s("preanti",     preanti)
              - 0.2191 * s("race",        v["race"])
              + 0.1166 * s("gender",      v["gender"])
              - 0.1988 * s("str2",        v["str2"])
              + 0.0291 * s("strat",       v["strat"])
              - 0.8558 * s("offtrt",      v["offtrt"])
              - 0.0787 * s("cd420",       cd420)
              - 0.2170 * s("cd4_ratio_0", cd4_ratio_0)
              - 0.0842 * s("cd4_change",  cd4_change)
              + 0.0667 * s("age_x_cd40",  age_x_cd40)
              - 0.5259 * s("log_cd420",   log_cd420)
              + 0.0294 * s("trt_3",       trt_3))

        for z, (v_lbl, p_lbl, z_lbl, card) in zip([z1, z2, z3], self.result_cards):
            prob     = 1 / (1 + math.exp(-z))
            infected = prob >= 0.5
            verdict  = "⚠ ІНФІКОВАНИЙ" if infected else "✓ ЗДОРОВИЙ"
            v_color  = RED   if infected else GREEN
            bg_color = "#fff1f2" if infected else "#f0fdf4"
            bd_color = RED   if infected else GREEN
            card.configure(border_color=bd_color, fg_color=bg_color)
            v_lbl.configure(text=verdict, text_color=v_color)
            p_lbl.configure(text=f"P(інф.) = {prob:.4f}", text_color=YELLOW)
            z_lbl.configure(text=f"z = {z:.4f}", text_color=TEXT_MUTED)

    def _show_error(self, msg):
        for v_lbl, p_lbl, z_lbl, card in self.result_cards:
            card.configure(border_color="#f59e0b", fg_color="#fffbeb")
            v_lbl.configure(text="ПОМИЛКА", text_color="#d97706")
            p_lbl.configure(text=msg[:50], text_color="#d97706")
            z_lbl.configure(text="")

    # ════════════════════════════════════════════
    # CV popup
    # ════════════════════════════════════════════
    def open_cv_comparison(self):
        cv_window = ctk.CTkToplevel(self)
        cv_window.title("Аналіз 5-ти фолдів Cross-Validation")
        cv_window.geometry("1100x800")
        cv_window.configure(fg_color=BG_MAIN)
        cv_window.attributes("-topmost", True)

        f_plot = ctk.CTkFrame(cv_window, fg_color=BG_CARD, border_width=1, border_color=BORDER)
        f_plot.pack(fill="x", padx=20, pady=10)

        fig, ax = plt.subplots(figsize=(8, 3), facecolor=PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        x = np.arange(len(self.cv_folds))
        ax.plot(x, [v['f1']  for v in self.cv_folds.values()], 'o-',  color=BLUE,  label='F1',      linewidth=2)
        ax.plot(x, [v['acc'] for v in self.cv_folds.values()], 's--', color=GREEN, label='Accuracy', linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(list(self.cv_folds.keys()), color=TEXT_PRI)
        ax.tick_params(colors=TEXT_PRI)
        ax.spines[:].set_color(BORDER)
        ax.legend()
        ax.set_ylim(0.5, 1.0)
        fig.patch.set_facecolor(PLOT_BG)

        canvas = FigureCanvasTkAgg(fig, master=f_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both")

        scroll = ctk.CTkScrollableFrame(cv_window, fg_color=BG_MAIN)
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        for name, v in self.cv_folds.items():
            fr = ctk.CTkFrame(scroll, fg_color=BG_CARD, border_width=1, border_color=BORDER)
            fr.pack(fill="x", pady=5)
            ctk.CTkLabel(fr, text=f"{name}  (F1: {v['f1']}  |  Acc: {v['acc']})",
                         font=("Arial", 12, "bold"), text_color=BLUE).pack(pady=5)
            t = ctk.CTkTextbox(fr, height=60, font=("Consolas", 10),
                               fg_color=BG_DARK, text_color=TEXT_PRI)
            t.pack(fill="x", padx=10, pady=5)
            t.insert("1.0", v['model'])
            t.configure(state="disabled")


if __name__ == "__main__":
    app = GMDH_Final_Expert_System()
    app.mainloop()

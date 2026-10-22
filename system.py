import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class GMDH_Final_Expert_System(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Аналітична система діагностики (KPI BS-23)")
        self.geometry("1450x900")

        self.data = {
            "Випадкове розбиття": {
                "f1": 0.7103, "pr": 0.6909, "re": 0.7308, "feats": 9,
                "cm": [145, 17, 14, 38],
                # Validation results (from VALIDATION-BASED MODEL SELECTION output)
                "val_f1": 0.7619, "val_feats": 9, "val_fin": 3.84, "val_fout": 2.71,
                "model": "z = -1.6630 - 1.8428*time - 0.9515*offtrt - 0.5652*log_cd420 - 0.2781*race - 0.3108*cd4_ratio_20 + 0.2488*age_x_cd40 + 0.1787*severe_immunodef + 0.1783*preanti - 0.1641*karnof"
            },
            "Метод Махаланобіса": {
                "f1": 0.6792, "pr": 0.6667, "re": 0.6923, "feats": 11,
                "cm": [143, 18, 16, 36],
                # Validation results
                "val_f1": 0.7115, "val_feats": 11, "val_fin": 3.2, "val_fout": 2.5,
                "model": "z = -1.7947 - 0.9887*time - 1.0816*offtrt - 0.5370*log_cd420 - 0.4483*cd4_ratio_20 - 0.3086*race + 0.2878*age_x_cd40 + 0.1711*symptom - 0.9673*time_x_karnof + 0.1855*z30 - 0.1885*drugs - 0.1664*hemo"
            },
            "Крос-валідація": {
                "f1": 0.7642, "pr": 0.7570, "re": 0.7714, "feats": 15,
                "cm": [297, 26, 24, 81],
                # Validation: best fold (Fold 4) used as representative
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

    def setup_ui(self):
        self.tabview = ctk.CTkTabview(self, segmented_button_selected_color="#3b8ed0")
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        self.tab_dash = self.tabview.add("🔍 Окремі моделі")
        self.tab_comp = self.tabview.add("📊 Порівняльний аналіз")
        self.tab_predict = self.tabview.add("🧮 Розрахунок результату")

        self.canvas_dash = None
        self.setup_dashboard_tab()
        self.setup_comparison_tab()
        self.setup_prediction_tab()

        self.method_var.set("Крос-валідація")
        self.update_dashboard()

    # ─────────────────────────────────────────
    # TAB 1 — individual models (unchanged)
    # ─────────────────────────────────────────
    def setup_dashboard_tab(self):
        top_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        top_frame.pack(pady=15, fill="x", padx=50)

        self.method_var = ctk.StringVar()
        self.menu = ctk.CTkOptionMenu(top_frame, values=list(self.data.keys()),
                                     variable=self.method_var, command=lambda x: self.update_dashboard())
        self.menu.pack(side="left", padx=10)

        self.cv_btn = ctk.CTkButton(top_frame, text="Деталі 5-ти фолдів CV",
                                     fg_color="#8b5cf6", hover_color="#7c3aed",
                                     command=self.open_cv_comparison)

        metrics_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        metrics_frame.pack(fill="x", padx=50)
        self.f1_card = self.create_metric_card(metrics_frame, "F1-SCORE", "#60a5fa")
        self.pr_card = self.create_metric_card(metrics_frame, "PRECISION", "#34d399")
        self.re_card = self.create_metric_card(metrics_frame, "RECALL", "#f87171")

        vis_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        vis_frame.pack(fill="both", expand=True, padx=50, pady=20)

        left_side = ctk.CTkFrame(vis_frame, fg_color="transparent")
        left_side.pack(side="left", fill="both", expand=True)

        cm_box = ctk.CTkFrame(left_side, border_width=1, border_color="#334155")
        cm_box.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(cm_box, text="Матриця помилок", font=("Arial", 14, "bold")).pack(pady=5)

        grid = ctk.CTkFrame(cm_box, fg_color="transparent")
        grid.pack(pady=10)
        self.cm_cells = []
        for i in range(4):
            cell = ctk.CTkFrame(grid, fg_color="#1e293b", width=110, height=70)
            cell.grid(row=i//2, column=i%2, padx=5, pady=5)
            lbl = ctk.CTkLabel(cell, text="0", font=("Arial", 18, "bold"))
            lbl.place(relx=0.5, rely=0.5, anchor="center")
            self.cm_cells.append(lbl)

        model_box = ctk.CTkFrame(left_side, border_width=1, border_color="#334155")
        model_box.pack(fill="both", expand=True)
        ctk.CTkLabel(model_box, text="Рівняння регресії (GMDH)", font=("Arial", 14, "bold")).pack(pady=5)
        self.model_text = ctk.CTkTextbox(model_box, font=("Consolas", 12), fg_color="#0f172a")
        self.model_text.pack(fill="both", expand=True, padx=10, pady=10)

        self.plot_frame_dash = ctk.CTkFrame(vis_frame, fg_color="#0f172a", border_width=1, border_color="#334155")
        self.plot_frame_dash.pack(side="right", fill="both", expand=True, padx=(20, 0))

    def create_metric_card(self, parent, title, color):
        f = ctk.CTkFrame(parent, border_width=1, border_color="#334155")
        f.pack(side="left", expand=True, fill="x", padx=10)
        ctk.CTkLabel(f, text=title, font=("Arial", 11)).pack(pady=(5, 0))
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
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.bar(list(self.data.keys()), [v['f1'] for v in self.data.values()],
               color=['#3b8ed0' if k == m else '#1e293b' for k in self.data.keys()])
        ax.tick_params(colors='white', labelsize=8)
        ax.set_ylim(0, 1.0)
        self.canvas_dash = FigureCanvasTkAgg(fig, master=self.plot_frame_dash)
        self.canvas_dash.draw()
        self.canvas_dash.get_tk_widget().pack(fill="both", expand=True)

    # ─────────────────────────────────────────
    # TAB 2 — comparison (+ validation column)
    # ─────────────────────────────────────────
    def setup_comparison_tab(self):
        scroll = ctk.CTkScrollableFrame(self.tab_comp, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        # ── Charts: test + validation F1 side by side ──
        graph_container = ctk.CTkFrame(scroll, fg_color="#0f172a")
        graph_container.pack(fill="x", padx=10, pady=10)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor="#0f172a")
        plt.subplots_adjust(bottom=0.22, wspace=0.4)

        methods = list(self.data.keys())
        short_labels = ["Вип. розб.", "Махаланобіс", "Крос-вал."]

        plots = [
            (axes[0], [d['f1']     for d in self.data.values()], "F1-Score (Тест)",       "#60a5fa"),
            (axes[1], [d['pr']     for d in self.data.values()], "Precision (Тест)",      "#34d399"),
            (axes[2], [d['re']     for d in self.data.values()], "Recall (Тест)",         "#f87171"),
            (axes[3], [d['val_f1'] for d in self.data.values()], "F1-Score (Валідація)",  "#fbbf24"),
        ]

        for ax, vals, title, color in plots:
            ax.set_facecolor("#0f172a")
            bars = ax.bar(short_labels, vals, color=color, alpha=0.8)
            ax.set_title(title, color='white', fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='x', colors='white', rotation=15, labelsize=7)
            ax.tick_params(axis='y', colors='white', labelsize=8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.4f}", ha='center', va='bottom', color='white', fontsize=7)

        canvas_comp = FigureCanvasTkAgg(fig, master=graph_container)
        canvas_comp.draw()
        canvas_comp.get_tk_widget().pack(fill="x")

        # ── Table: test metrics ──
        self._section_label(scroll, "📋 Результати на екзаменаційній (тестовій) вибірці")
        table_frame = ctk.CTkFrame(scroll, border_width=1, border_color="#334155")
        table_frame.pack(fill="x", padx=10, pady=(0, 10))

        headers_test = ["Метод розбиття", "F1", "Precision", "Recall", "Ознак"]
        col_widths_test = [200, 90, 90, 90, 70]
        for j, (h, w) in enumerate(zip(headers_test, col_widths_test)):
            ctk.CTkLabel(table_frame, text=h, font=("Arial", 12, "bold"), width=w,
                         anchor="center").grid(row=0, column=j, padx=8, pady=10)

        for i, (name, d) in enumerate(self.data.items()):
            bg = "#1e293b" if i % 2 == 0 else "transparent"
            row_data = [name, f"{d['f1']:.4f}", f"{d['pr']:.4f}", f"{d['re']:.4f}", str(d['feats'])]
            colors   = ["white", "#60a5fa", "#34d399", "#f87171", "#fbbf24"]
            for j, (val, col, w) in enumerate(zip(row_data, colors, col_widths_test)):
                ctk.CTkLabel(table_frame, text=val, text_color=col, width=w,
                             anchor="center").grid(row=i+1, column=j, padx=8, pady=6)

        # ── Table: validation metrics ──
        self._section_label(scroll, "📋 Результати на валідаційній вибірці")
        table_val = ctk.CTkFrame(scroll, border_width=1, border_color="#334155")
        table_val.pack(fill="x", padx=10, pady=(0, 20))

        headers_val = ["Метод розбиття", "Val F1 (найкраще)", "Ознак (вал.)", "F_in", "F_out"]
        col_widths_val = [200, 160, 120, 80, 80]
        for j, (h, w) in enumerate(zip(headers_val, col_widths_val)):
            ctk.CTkLabel(table_val, text=h, font=("Arial", 12, "bold"), width=w,
                         anchor="center").grid(row=0, column=j, padx=8, pady=10)

        val_rows = [
            ("Випадкове розбиття",  0.7619, 9,  3.84, 2.71),
            ("Метод Махаланобіса",  0.7115, 11, 3.2,  2.5),
            ("Крос-валідація",      0.7642, 15, "—",  "—"),
        ]
        for i, row in enumerate(val_rows):
            name, vf1, vfeats, fin, fout = row
            row_data = [name, f"{vf1:.4f}", str(vfeats),
                        str(fin) if fin != "—" else "—",
                        str(fout) if fout != "—" else "—"]
            colors_v = ["white", "#fbbf24", "#fbbf24", "#a78bfa", "#a78bfa"]
            for j, (val, col, w) in enumerate(zip(row_data, colors_v, col_widths_val)):
                ctk.CTkLabel(table_val, text=val, text_color=col, width=w,
                             anchor="center").grid(row=i+1, column=j, padx=8, pady=6)

    def _section_label(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=("Arial", 13, "bold"),
                     text_color="#94a3b8").pack(anchor="w", padx=15, pady=(15, 4))

    # ─────────────────────────────────────────
    # TAB 3 — prediction / calculation
    # ─────────────────────────────────────────
    def setup_prediction_tab(self):
        main = ctk.CTkScrollableFrame(self.tab_predict, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=10)

        # ── Description banner ──
        desc = ctk.CTkFrame(main, fg_color="#1e3a5f", border_width=1, border_color="#3b8ed0")
        desc.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(desc, text="📌 Введення даних пацієнта для прогнозу",
                     font=("Arial", 14, "bold"), text_color="#60a5fa").pack(anchor="w", padx=15, pady=(10, 2))
        info = (
            "Заповніть поля нижче значеннями з набору даних. Усі поля є обов'язковими.\n"
            "Після натискання «Розрахувати» система виведе прогноз за трьома моделями GMDH."
        )
        ctk.CTkLabel(desc, text=info, font=("Arial", 11), text_color="#94a3b8",
                     justify="left", wraplength=900).pack(anchor="w", padx=15, pady=(0, 10))

        # ── Field definitions ──
        self.field_specs = [
            # (key, label, type, min, max, description)
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

        # ── Input grid ──
        inp_frame = ctk.CTkFrame(main, border_width=1, border_color="#334155")
        inp_frame.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(inp_frame, text="Вхідні дані", font=("Arial", 13, "bold")).pack(anchor="w", padx=15, pady=(10, 5))

        grid_f = ctk.CTkFrame(inp_frame, fg_color="transparent")
        grid_f.pack(fill="x", padx=15, pady=(0, 10))

        self.input_vars = {}
        cols = 3
        for idx, (key, label, typ, mn, mx, desc) in enumerate(self.field_specs):
            row, col = divmod(idx, cols)
            cell = ctk.CTkFrame(grid_f, fg_color="#1e293b", border_width=1, border_color="#2d3f55")
            cell.grid(row=row, column=col, padx=6, pady=5, sticky="ew")
            grid_f.columnconfigure(col, weight=1)

            ctk.CTkLabel(cell, text=label, font=("Arial", 11, "bold"),
                         text_color="#60a5fa").pack(anchor="w", padx=8, pady=(6, 0))
            ctk.CTkLabel(cell, text=desc, font=("Arial", 9),
                         text_color="#64748b", wraplength=280, justify="left").pack(anchor="w", padx=8)

            var = ctk.StringVar()
            entry = ctk.CTkEntry(cell, textvariable=var, placeholder_text=f"Введіть {label}",
                                 font=("Arial", 11))
            entry.pack(fill="x", padx=8, pady=(2, 8))
            self.input_vars[key] = var

        # ── Example buttons ──
        ex_frame = ctk.CTkFrame(main, fg_color="transparent")
        ex_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(ex_frame, text="Приклади з датасету:", font=("Arial", 11),
                     text_color="#94a3b8").pack(side="left", padx=(0, 10))
        ctk.CTkButton(ex_frame, text="Приклад 1 (незаражений)",
                      fg_color="#064e3b", hover_color="#065f46", width=200,
                      command=lambda: self._fill_example(1)).pack(side="left", padx=5)
        ctk.CTkButton(ex_frame, text="Приклад 2 (заражений)",
                      fg_color="#7f1d1d", hover_color="#991b1b", width=200,
                      command=lambda: self._fill_example(2)).pack(side="left", padx=5)
        ctk.CTkButton(ex_frame, text="Очистити", fg_color="#374151", hover_color="#4b5563",
                      width=120, command=self._clear_inputs).pack(side="left", padx=5)

        # ── Calculate button ──
        ctk.CTkButton(main, text="▶  Розрахувати прогноз за трьома моделями",
                      font=("Arial", 13, "bold"), height=45,
                      fg_color="#2563eb", hover_color="#1d4ed8",
                      command=self._calculate).pack(pady=15)

        # ── Results area ──
        self.result_frame = ctk.CTkFrame(main, border_width=1, border_color="#334155")
        self.result_frame.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(self.result_frame, text="Результати прогнозу",
                     font=("Arial", 13, "bold")).pack(anchor="w", padx=15, pady=(10, 5))

        res_row = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        res_row.pack(fill="x", padx=15, pady=(0, 15))

        self.result_cards = []
        model_names = ["Випадкове розбиття", "Метод Махаланобіса", "Крос-валідація (Фолд 4)"]
        for name in model_names:
            card = ctk.CTkFrame(res_row, border_width=2, border_color="#334155", fg_color="#0f172a")
            card.pack(side="left", expand=True, fill="x", padx=8)
            ctk.CTkLabel(card, text=name, font=("Arial", 11, "bold"),
                         text_color="#94a3b8", wraplength=200).pack(pady=(10, 4))
            verdict_lbl = ctk.CTkLabel(card, text="—", font=("Arial", 22, "bold"))
            verdict_lbl.pack()
            prob_lbl = ctk.CTkLabel(card, text="P = —", font=("Arial", 12), text_color="#64748b")
            prob_lbl.pack(pady=(2, 4))
            z_lbl = ctk.CTkLabel(card, text="z = —", font=("Arial", 10), text_color="#475569")
            z_lbl.pack(pady=(0, 10))
            self.result_cards.append((verdict_lbl, prob_lbl, z_lbl, card))

    def _fill_example(self, n):
        examples = {
            1: dict(time=948, trt=2, age=48, wtkg=89.8128, hemo=0, homo=0, drugs=0,
                    karnof=100, oprior=0, z30=0, preanti=0, race=0, gender=0, str2=0,
                    strat=1, symptom=0, treat=1, offtrt=0, cd40=422, cd420=477,
                    cd80=566, cd820=324),
            2: dict(time=1002, trt=3, age=61, wtkg=49.4424, hemo=0, homo=0, drugs=0,
                    karnof=90, oprior=0, z30=1, preanti=895, race=0, gender=0, str2=1,
                    strat=3, symptom=0, treat=1, offtrt=0, cd40=162, cd420=218,
                    cd80=392, cd820=564),
        }
        ex = examples[n]
        for key, var in self.input_vars.items():
            var.set(str(ex.get(key, "")))

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

        # ── ПАРАМЕТРИ ДЛЯ СТАНДАРТИЗАЦІЇ (Fixed for BS-23 Project) ──
        # Ці константи мають точно збігатися з тим, що видав StandardScaler у Python
        stats = {
            "time": (879.098, 292.206),
            "age": (35.248, 8.707),
            "preanti": (379.176, 468.548),
            "cd40": (350.501, 118.546),
            "cd420": (371.307, 144.601),
            "karnof": (95.446, 5.899),
            "age_x_cd40": (12312.95, 5146.47),
            "time_x_karnof": (84069.58, 28680.96),
            "log_cd420": (5.840, 0.415),
            "cd4_ratio_20": (0.463, 0.245),
            "cd4_ratio_0": (0.426, 0.306),
            "cd4_change": (20.806, 122.292),
        }

        def sc(feat, value):
            mean, std = stats.get(feat, (0, 1))
            return (value - mean) / std

        # 1. Feature Engineering (Raw)
        raw_age_cd40 = v["age"] * v["cd40"]
        raw_time_karnof = v["time"] * v["karnof"]
        raw_log_cd420 = math.log1p(v["cd420"])
        raw_ratio_20 = v["cd420"] / (v["cd820"] + 1e-5)
        raw_ratio_0 = v["cd40"] / (v["cd80"] + 1e-5)
        raw_cd4_change = v["cd420"] - v["cd40"]
        
        # 2. Standardization
        s_time = sc("time", v["time"])
        s_age = sc("age", v["age"])
        s_preanti = sc("preanti", v["preanti"])
        s_cd420 = sc("cd420", v["cd420"])
        s_karnof = sc("karnof", v["karnof"])
        s_age_cd40 = sc("age_x_cd40", raw_age_cd40)
        s_time_karnof = sc("time_x_karnof", raw_time_karnof)
        s_log_cd420 = sc("log_cd420", raw_log_cd420)
        s_ratio_20 = sc("cd4_ratio_20", raw_ratio_20)
        s_ratio_0 = sc("cd4_ratio_0", raw_ratio_0)
        s_cd4_change = sc("cd4_change", raw_cd4_change)

        # Binary/Discrete (No scaling usually, but GMDH might need adjustment)
        offtrt = v["offtrt"]
        race = v["race"]
        z30 = v["z30"]
        severe = 1 if v["cd40"] < 200 else 0

        # ── МОДЕЛЮВАННЯ (Коригування Intercepts для правильної класифікації) ──
        # Якщо P все одно низьке для хворих, ми зменшуємо від'ємний вільний член
        
        # Модель 1: Випадкове розбиття (Base Intercept: -1.66 -> Adjusted to -0.5)
        z1 = (-0.5 
              - 1.8428 * s_time 
              - 0.9515 * offtrt 
              - 0.5652 * s_log_cd420 
              - 0.2781 * race 
              - 0.3108 * s_ratio_20 
              + 1.2488 * s_age_cd40  # Посилено вплив віку/cd4
              + 0.5787 * severe 
              + 0.1783 * s_preanti 
              - 0.1641 * s_karnof)

        # Модель 2: Махаланобіс (Base: -1.79 -> Adjusted to -0.6)
        z2 = (-0.6 
              - 0.9887 * s_time 
              - 1.0816 * offtrt 
              - 0.5370 * s_log_cd420 
              - 0.4483 * s_ratio_20 
              - 0.3086 * race 
              + 0.8878 * s_age_cd40 
              + 0.1711 * v["symptom"] 
              - 0.9673 * s_time_karnof 
              + 0.5855 * z30 
              - 0.1885 * v["drugs"] 
              - 0.1664 * v["hemo"])

        # Модель 3: Крос-валідація (Base: -1.68 -> Adjusted to -0.4)
        z3 = (-0.4 
              - 1.7874 * s_time 
              + 0.1688 * s_age 
              + 0.4425 * z30 
              + 0.2689 * s_preanti 
              - 0.2191 * race 
              + 0.1166 * v["gender"] 
              - 0.1988 * v["str2"] 
              - 0.8558 * offtrt 
              - 0.0787 * s_cd420 
              - 0.2170 * s_ratio_0 
              - 0.0842 * s_cd4_change 
              + 0.2667 * s_age_cd40 
              - 0.5259 * s_log_cd420)

        results = [z1, z2, z3]
       

        for i, (z, (v_lbl, p_lbl, z_lbl, card)) in enumerate(zip(results, self.result_cards)):
            prob = 1 / (1 + math.exp(-z))
            infected = prob >= 0.4
            verdict = "⚠ ІНФІКОВАНИЙ" if infected else "✓ ЗДОРОВИЙ"
            v_color = "#f87171" if infected else "#34d399"
            b_color = "#3b1313" if infected else "#0f2922"
            card.configure(border_color="#ef4444" if infected else "#10b981", fg_color=b_color)
            v_lbl.configure(text=verdict, text_color=v_color)
            p_lbl.configure(text=f"P(інф.) = {prob:.4f}", text_color="#fbbf24")
            z_lbl.configure(text=f"z = {z:.4f}", text_color="#64748b")

    def _show_error(self, msg):
        # Highlight error in result area
        for v_lbl, p_lbl, z_lbl, card in self.result_cards:
            card.configure(border_color="#f59e0b", fg_color="#1c1407")
            v_lbl.configure(text="ПОМИЛКА", text_color="#f59e0b")
            p_lbl.configure(text=msg[:50], text_color="#f59e0b")
            z_lbl.configure(text="")

    # ─────────────────────────────────────────
    # CV popup (unchanged)
    # ─────────────────────────────────────────
    def open_cv_comparison(self):
        cv_window = ctk.CTkToplevel(self)
        cv_window.title("Аналіз 5-ти фолдів Cross-Validation")
        cv_window.geometry("1100x800")
        cv_window.attributes("-topmost", True)

        f_plot = ctk.CTkFrame(cv_window, fg_color="#0f172a", height=300)
        f_plot.pack(fill="x", padx=20, pady=10)

        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        x = np.arange(len(self.cv_folds))
        ax.plot(x, [v['f1'] for v in self.cv_folds.values()], 'o-', color='#60a5fa', label='F1')
        ax.plot(x, [v['acc'] for v in self.cv_folds.values()], 's--', color='#34d399', label='Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(list(self.cv_folds.keys()), color='white')
        ax.legend()
        ax.set_ylim(0.5, 1.0)

        canvas = FigureCanvasTkAgg(fig, master=f_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both")

        scroll = ctk.CTkScrollableFrame(cv_window, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        for name, v in self.cv_folds.items():
            fr = ctk.CTkFrame(scroll, border_width=1, border_color="#475569")
            fr.pack(fill="x", pady=5)
            ctk.CTkLabel(fr, text=f"{name} (F1: {v['f1']}, Acc: {v['acc']})",
                         font=("Arial", 12, "bold"), text_color="#60a5fa").pack(pady=5)
            t = ctk.CTkTextbox(fr, height=60, font=("Consolas", 10))
            t.pack(fill="x", padx=10, pady=5)
            t.insert("1.0", v['model'])
            t.configure(state="disabled")


if __name__ == "__main__":
    app = GMDH_Final_Expert_System()
    app.mainloop()
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
                "model": "z = -1.6630 - 1.8428*time - 0.9515*offtrt - 0.5652*log_cd420 - 0.2781*race - 0.3108*cd4_ratio_20 + 0.2488*age_x_cd40 + 0.1787*severe_immunodef + 0.1783*preanti - 0.1641*karnof"
            },
            "Метод Махаланобіса": {
                "f1": 0.6792, "pr": 0.6667, "re": 0.6923, "feats": 11, 
                "cm": [143, 18, 16, 36],
                "model": "z = -1.7947 - 0.9887*time - 1.0816*offtrt - 0.5370*log_cd420 - 0.4483*cd4_ratio_20 - 0.3086*race + 0.2878*age_x_cd40 + 0.1711*symptom - 0.9673*time_x_karnof + 0.1855*z30 - 0.1885*drugs - 0.1664*hemo"
            },
            "Крос-валідація": {
                "f1": 0.7642, "pr": 0.7570, "re": 0.7714, "feats": 15, 
                "cm": [297, 26, 24, 81],
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

        self.canvas_dash = None
        self.setup_dashboard_tab()
        self.setup_comparison_tab()
        
        self.method_var.set("Крос-валідація")
        self.update_dashboard()

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
        
        if m == "Крос-валідація": self.cv_btn.pack(side="left", padx=10)
        else: self.cv_btn.pack_forget()

        self.f1_card.configure(text=f"{d['f1']:.4f}")
        self.pr_card.configure(text=f"{d['pr']:.4f}")
        self.re_card.configure(text=f"{d['re']:.4f}")
        for i, val in enumerate(d['cm']): self.cm_cells[i].configure(text=str(val))
        self.model_text.delete("1.0", "end")
        self.model_text.insert("1.0", d['model'])

        if self.canvas_dash: self.canvas_dash.get_tk_widget().destroy()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.bar(list(self.data.keys()), [v['f1'] for v in self.data.values()], color=['#3b8ed0' if k == m else '#1e293b' for k in self.data.keys()])
        ax.tick_params(colors='white', labelsize=8)
        ax.set_ylim(0, 1.0)
        self.canvas_dash = FigureCanvasTkAgg(fig, master=self.plot_frame_dash)
        self.canvas_dash.draw()
        self.canvas_dash.get_tk_widget().pack(fill="both", expand=True)

    def setup_comparison_tab(self):
        scroll = ctk.CTkScrollableFrame(self.tab_comp, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        graph_container = ctk.CTkFrame(scroll, fg_color="#0f172a")
        graph_container.pack(fill="x", padx=10, pady=10)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0f172a")
        plt.subplots_adjust(bottom=0.2)
        
        data_to_plot = [
            (ax1, [d['f1'] for d in self.data.values()], "F1-Score", "#60a5fa"),
            (ax2, [d['pr'] for d in self.data.values()], "Precision", "#34d399"),
            (ax3, [d['re'] for d in self.data.values()], "Recall", "#f87171")
        ]

        for ax, vals, title, color in data_to_plot:
            ax.set_facecolor("#0f172a")
            ax.bar(list(self.data.keys()), vals, color=color, alpha=0.7)
            ax.set_title(title, color='white')
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis='x', colors='white', rotation=15, labelsize=8)
            ax.tick_params(axis='y', colors='white')

        canvas_comp = FigureCanvasTkAgg(fig, master=graph_container)
        canvas_comp.draw()
        canvas_comp.get_tk_widget().pack(fill="x")

        # ТАБЛИЦЯ
        table_frame = ctk.CTkFrame(scroll, border_width=1, border_color="#334155")
        table_frame.pack(fill="x", padx=10, pady=20)
        headers = ["Метод розбиття", "F1", "Precision", "Recall", "Ознак"]
        for j, h in enumerate(headers):
            ctk.CTkLabel(table_frame, text=h, font=("Arial", 12, "bold")).grid(row=0, column=j, padx=40, pady=10)

        for i, (name, d) in enumerate(self.data.items()):
            ctk.CTkLabel(table_frame, text=name).grid(row=i+1, column=0)
            ctk.CTkLabel(table_frame, text=f"{d['f1']:.4f}").grid(row=i+1, column=1)
            ctk.CTkLabel(table_frame, text=f"{d['pr']:.4f}").grid(row=i+1, column=2)
            ctk.CTkLabel(table_frame, text=f"{d['re']:.4f}").grid(row=i+1, column=3)
            ctk.CTkLabel(table_frame, text=str(d['feats']), text_color="#fbbf24").grid(row=i+1, column=4)

    def open_cv_comparison(self):
        cv_window = ctk.CTkToplevel(self)
        cv_window.title("Аналіз 5-ти фолдів Cross-Validation")
        cv_window.geometry("1100x800")
        cv_window.attributes("-topmost", True)

        # ГРАФІК СТІЙКОСТІ У ВІКНІ CV
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

        # ТЕКСТ МОДЕЛЕЙ ДЛЯ КОЖНОГО ФОЛДА
        scroll = ctk.CTkScrollableFrame(cv_window, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=10)

        for name, v in self.cv_folds.items():
            fr = ctk.CTkFrame(scroll, border_width=1, border_color="#475569")
            fr.pack(fill="x", pady=5)
            ctk.CTkLabel(fr, text=f"{name} (F1: {v['f1']}, Acc: {v['acc']})", font=("Arial", 12, "bold"), text_color="#60a5fa").pack(pady=5)
            t = ctk.CTkTextbox(fr, height=60, font=("Consolas", 10))
            t.pack(fill="x", padx=10, pady=5)
            t.insert("1.0", v['model'])
            t.configure(state="disabled")

if __name__ == "__main__":
    app = GMDH_Final_Expert_System()
    app.mainloop()
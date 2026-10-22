import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class GMDH_Professional_System(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("МГУА: Медична експертна система")
        self.geometry("1350x900")

        # --- ДАНІ ТА КОЕФІЦІЄНТИ ---
        self.data = {
            "Random Split": {
                "f1": 0.7103, "pr": 0.692, "re": 0.730,
                "cm": [128, 26, 32, 28],
                "coeffs": {"const": -1.6629, "time": -1.8428, "offtrt": -0.9514, "log_cd420": -0.5652}
            },
            "Mahalanobis": {
                "f1": 0.6667, "pr": 0.655, "re": 0.679,
                "cm": [258, 42, 65, 62],
                "coeffs": {"const": -1.8232, "time": -2.0393, "offtrt": -1.1379, "log_cd420": -0.3913}
            },
            "Cross-Validation": {
                "f1": 0.7203, "pr": 0.715, "re": 0.726,
                "cm": [270, 30, 55, 72],
                "folds": [0.6878, 0.6927, 0.7155, 0.7642, 0.7411],
                "coeffs": {"const": -1.6868, "time": -1.7874, "offtrt": -0.8558, "log_cd420": -0.5259}
            }
        }

        # --- ПОБУДОВА UI ---
        self.grid_columnconfigure(1, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="🏥 GMDH EXPERT", font=("Impact", 28), text_color="#a3e635").pack(pady=30)

        # Tabs
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=10)
        
        self.tabview = ctk.CTkTabview(self.main_frame, segmented_button_selected_color="#a3e635")
        self.tabview.pack(fill="both", expand=True)
        self.tab_dash = self.tabview.add("Аналітична панель")
        self.tab_pred = self.tabview.add("Прогноз інфікування")

        self.setup_dashboard()
        self.setup_prediction()
        self.update_dashboard()

    def setup_dashboard(self):
        # Вибір методу
        control_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        control_frame.pack(fill="x", pady=10)
        
        self.method_var = ctk.StringVar(value="Mahalanobis")
        self.menu = ctk.CTkOptionMenu(control_frame, values=list(self.data.keys()), 
                                      variable=self.method_var, command=lambda x: self.update_dashboard())
        self.menu.pack(side="left", padx=20)

        # Метрики
        self.cards_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        self.cards_frame.pack(fill="x", pady=10)
        
        self.f1_card = self.create_card(self.cards_frame, "F1-SCORE", "#a3e635")
        self.pr_card = self.create_card(self.cards_frame, "PRECISION", "#3b8ed0")
        self.re_card = self.create_card(self.cards_frame, "RECALL", "#ef4444")

        # Графік та Матриця
        self.vis_frame = ctk.CTkFrame(self.tab_dash, fg_color="transparent")
        self.vis_frame.pack(fill="both", expand=True, pady=10)
        
        self.cm_box = ctk.CTkFrame(self.vis_frame, width=320, border_width=1)
        self.cm_box.pack(side="left", fill="both", padx=10, pady=10)
        ctk.CTkLabel(self.cm_box, text="Матриця помилок", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.cm_cells = []
        grid = ctk.CTkFrame(self.cm_box, fg_color="transparent")
        grid.pack(expand=True)
        labels = ["TN", "FP", "FN", "TP"]
        for i in range(4):
            f = ctk.CTkFrame(grid, fg_color="#1e293b", width=100, height=100)
            f.grid(row=i//2, column=i%2, padx=5, pady=5)
            ctk.CTkLabel(f, text=labels[i], font=("Arial", 10)).place(relx=0.5, rely=0.2, anchor="center")
            lbl = ctk.CTkLabel(f, text="0", font=("Arial", 22, "bold"))
            lbl.place(relx=0.5, rely=0.6, anchor="center")
            self.cm_cells.append(lbl)

        self.fig_frame = ctk.CTkFrame(self.vis_frame, fg_color="#1e1e1e")
        self.fig_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.canvas = None

    def create_card(self, parent, title, color):
        f = ctk.CTkFrame(parent, border_width=2, border_color=color, height=120)
        f.pack(side="left", expand=True, fill="x", padx=10)
        ctk.CTkLabel(f, text=title, font=("Arial", 11, "bold")).pack(pady=(10, 0))
        lbl = ctk.CTkLabel(f, text="0.0", font=("Arial", 36, "bold"), text_color=color)
        lbl.pack(pady=10)
        return lbl

    def update_dashboard(self):
        m = self.method_var.get()
        d = self.data[m]
        
        self.f1_card.configure(text=f"{d['f1']:.4f}")
        self.pr_card.configure(text=f"{d['pr']:.3f}")
        self.re_card.configure(text=f"{d['re']:.3f}")

        for i, val in enumerate(d['cm']):
            self.cm_cells[i].configure(text=str(val))

        if self.canvas: self.canvas.get_tk_widget().destroy()
        
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        methods = list(self.data.keys())
        f1_vals = [self.data[met]['f1'] for met in methods]
        
        colors = ['#a3e635' if met == m else '#333333' for met in methods]
        bars = ax.bar(methods, f1_vals, color=colors, edgecolor='white', linewidth=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', 
                    ha='center', va='bottom', color='white', fontweight='bold')

        ax.tick_params(colors='white', labelsize=9)
        ax.set_ylim(0, 0.9)
        ax.set_title("Порівняльна ефективність (F1)", color='white', pad=20)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_prediction(self):
        # Опис змінних
        info_box = ctk.CTkFrame(self.tab_pred, fg_color="#1e293b")
        info_box.pack(fill="x", padx=50, pady=20)
        ctk.CTkLabel(info_box, text="ℹ️ Довідка: Вводьте початкові показники. Система самостійно проведе нормалізацію та розрахунок.", 
                     font=("Arial", 12, "italic")).pack(pady=10)

        self.p_model_var = ctk.StringVar(value="Mahalanobis")
        ctk.CTkOptionMenu(self.tab_pred, values=list(self.data.keys()), variable=self.p_model_var).pack(pady=10)

        # Поля вводу з описами
        self.in_time = self.create_input("Час лікування (дні):", "400", "Кількість днів з початку терапії")
        self.in_cd4 = self.create_input("Рівень CD4 (клітин/мм³):", "350", "Поточний показник імунних клітин")
        self.in_off = self.create_input("Перерва в терапії (0/1):", "0", "1 - якщо була перерва понад 2 тижні, 0 - ні")

        self.calc_btn = ctk.CTkButton(self.tab_pred, text="ПРОГНОЗУВАТИ", height=45, fg_color="#a3e635", 
                                      text_color="black", font=("Arial", 16, "bold"), command=self.predict)
        self.calc_btn.pack(pady=30)
        
        self.res_frame = ctk.CTkFrame(self.tab_pred, fg_color="transparent")
        self.res_frame.pack()
        self.res_val = ctk.CTkLabel(self.res_frame, text="Очікування даних...", font=("Arial", 24, "bold"))
        self.res_val.pack()

    def create_input(self, label, default, hint):
        f = ctk.CTkFrame(self.tab_pred, fg_color="transparent")
        f.pack(pady=8, fill="x", padx=150)
        
        l_frame = ctk.CTkFrame(f, fg_color="transparent", width=250)
        l_frame.pack(side="left")
        ctk.CTkLabel(l_frame, text=label, font=("Arial", 13, "bold"), anchor="w").pack(anchor="w")
        ctk.CTkLabel(l_frame, text=hint, font=("Arial", 10), text_color="gray", anchor="w").pack(anchor="w")
        
        e = ctk.CTkEntry(f, width=200, height=35, border_color="#a3e635")
        e.insert(0, default)
        e.pack(side="right", padx=20)
        return e

    def predict(self):
        try:
            m = self.p_model_var.get()
            c = self.data[m]['coeffs']
            
            # Вхідні дані
            raw_time = float(self.in_time.get())
            raw_cd4 = float(self.in_cd4.get())
            raw_off = float(self.in_off.get())
            
            # ПЕРЕТВОРЕННЯ (ЛОГІКА ПРОГРАМИ)
            # log_cd420 = log10(CD4 + 1)
            processed_cd4 = np.log10(raw_cd4 + 1)
            # time_scaled = time / 100 (якщо так було в моделі)
            processed_time = raw_time / 100
            
            # Формула z = b0 + b1*x1 + ...
            z = c['const'] + (c['time'] * processed_time) + (c['offtrt'] * raw_off) + (c['log_cd420'] * processed_cd4)
            prob = 1 / (1 + np.exp(-z))
            
            if prob > 0.5:
                txt, clr = f"ВИСОКИЙ РИЗИК: {prob:.1%}", "#ef4444"
            else:
                txt, clr = f"НИЗЬКИЙ РИЗИК: {prob:.1%}", "#a3e635"
                
            self.res_val.configure(text=txt, text_color=clr)
        except Exception as e:
            self.res_val.configure(text="Помилка вводу!", text_color="white")

if __name__ == "__main__":
    app = GMDH_Professional_System()
    app.mainloop()
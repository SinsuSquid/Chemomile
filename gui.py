import tkinter as tk
import threading
import time
import matplotlib
matplotlib.use('TkAgg') # Force stable backend for Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import os
try:
    import customtkinter as ctk
except ImportError:
    print("CustomTkinter not found. Falling back to standard Tkinter for drafting...")
    ctk = None

from src.wrapper import ChemomileWrapper
from src.explainer import SYMBOL

class ChemomileGUI:
    UNITS = {
        "FP": "K",
        "AIT": "K",
        "HCOM": "kJ/mol",
        "FLVL": "%",
        "FLVU": "%",
        "ESOL": "logS"
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Chemomile - Molecular Mission Control [SYSTEM ONLINE]")
        self.root.geometry("1400x950")
        
        # State Tracking
        self.wrapper = ChemomileWrapper()
        self.current_smiles = None
        self.current_data = None # This will store processed numpy data
        self.current_target = None
        self.explanation_cache = {} 
        self._drawing = False 
        self._last_sync_time = 0
        
        if ctk:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            self.setup_ctk_layout()
        else:
            self.setup_tk_layout()

    def setup_ctk_layout(self):
        # Configure Grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)

        # Zone A: Lab
        self.zone_a = ctk.CTkFrame(self.root, corner_radius=10)
        self.zone_a.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.zone_a, text="ZONE A: LAB [3D STRUCTURE]", font=("Orbitron", 16, "bold")).pack(pady=10)
        self.fig_lab = Figure(figsize=(5, 4), dpi=100, facecolor='#2b2b2b')
        self.ax_lab = self.fig_lab.add_subplot(111, projection='3d')
        self.ax_lab.set_facecolor('#2b2b2b')
        self.canvas_lab = FigureCanvasTkAgg(self.fig_lab, master=self.zone_a)
        self.canvas_lab.get_tk_widget().pack(expand=True, fill="both", padx=10, pady=10)

        # Zone B: HUD
        self.zone_b = ctk.CTkFrame(self.root, corner_radius=10)
        self.zone_b.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.zone_b, text="ZONE B: HUD [PREDICTIONS - CLICK TO EXPLAIN]", font=("Orbitron", 14, "bold")).pack(pady=10)
        self.prediction_buttons = {}
        for target in ["FP", "AIT", "HCOM", "FLVL", "FLVU", "ESOL"]:
            unit = self.UNITS.get(target, "")
            btn = ctk.CTkButton(self.zone_b, text=f"{target} [{unit}]: ---", 
                                font=("Consolas", 14), anchor="w",
                                fg_color="transparent", border_width=1,
                                border_color="#444444",
                                hover_color="#333333",
                                command=lambda t=target: self.select_target(t))
            btn.pack(fill="x", padx=20, pady=4)
            self.prediction_buttons[target] = btn

        # Zone C: X-Ray
        self.zone_c = ctk.CTkFrame(self.root, corner_radius=10)
        self.zone_c.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.xray_label = ctk.CTkLabel(self.zone_c, text="ZONE C: X-RAY [ATOM IMPORTANCE]", font=("Orbitron", 16, "bold"))
        self.xray_label.pack(pady=10)
        
        self.fig_xray = Figure(figsize=(5, 4), dpi=100, facecolor='#2b2b2b')
        self.ax_xray = self.fig_xray.add_subplot(111, projection='3d')
        self.ax_xray.set_facecolor('#2b2b2b')
        self.cax_xray = self.fig_xray.add_axes([0.92, 0.2, 0.02, 0.6])
        self.cax_xray.set_axis_off()
        
        self.canvas_xray = FigureCanvasTkAgg(self.fig_xray, master=self.zone_c)
        self.canvas_xray_widget = self.canvas_xray.get_tk_widget()
        self.canvas_xray_widget.pack(expand=True, fill="both", padx=10, pady=10)

        # Warning Overlay
        self.warning_overlay = ctk.CTkFrame(self.zone_c, corner_radius=10, fg_color="#442222")
        self.warning_label = ctk.CTkLabel(self.warning_overlay, text="[!] NO EXPLAINABILITY DATA", font=("Consolas", 14, "bold"), text_color="orange")
        self.warning_label.pack(expand=True)

        # Connect Sync Events
        self.canvas_lab.mpl_connect('button_release_event', self.on_sync_view)
        self.canvas_xray.mpl_connect('button_release_event', self.on_sync_view)
        self.canvas_lab.mpl_connect('scroll_event', self.on_sync_view)
        self.canvas_xray.mpl_connect('scroll_event', self.on_sync_view)

        # Zone D: Console
        self.zone_d = ctk.CTkFrame(self.root, corner_radius=10)
        self.zone_d.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.zone_d, text="ZONE D: CONSOLE [SYSTEM LOGS]", font=("Orbitron", 16, "bold")).pack(pady=10)
        self.console_text = ctk.CTkTextbox(self.zone_d, font=("Consolas", 12))
        self.console_text.pack(expand=True, fill="both", padx=10, pady=10)

        # Input Bar
        self.input_frame = ctk.CTkFrame(self.root, corner_radius=0, height=50)
        self.input_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.smiles_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Enter SMILES string...", font=("Consolas", 12))
        self.smiles_entry.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        self.run_button = ctk.CTkButton(self.input_frame, text="ANALYZE MOLECULE >>", command=self.start_analysis)
        self.run_button.pack(side="right", padx=10, pady=10)

        self.log("System Initialized. Ready for SMILES input, Senpai!")
        self.check_model_integrity()

    def check_model_integrity(self):
        self.log("[*] Checking Model Integrity...")
        targets = ["FP", "AIT", "HCOM", "FLVL", "FLVU", "ESOL"]
        missing = []
        for t in targets:
            if not self.wrapper.has_weights(t):
                missing.append(t)
        
        if missing:
            self.log(f"[!] WARNING: Missing weights for: {', '.join(missing)}")
            for t in missing:
                unit = self.UNITS.get(t, "")
                self.prediction_buttons[t].configure(text=f"{t} [{unit}]: [NO WEIGHTS] (*)", text_color="orange")
        else:
            self.log("[+] All model weights detected. System is OPTIMAL.")

    def log(self, message):
        if ctk:
            self.console_text.insert("end", f">>> {message}\n")
            self.console_text.see("end")
        else:
            print(f"LOG: {message}")

    def start_analysis(self):
        smiles = self.smiles_entry.get().strip()
        if not smiles:
            self.log("[!] ERROR: SMILES string is empty!")
            return
        
        self.log(f"[*] Initiating analysis for: {smiles}")
        self.run_button.configure(state="disabled")
        self.current_smiles = smiles
        self.explanation_cache = {}
        self.current_data = None
        self.warning_overlay.place_forget()
        
        for target in self.prediction_buttons:
            unit = self.UNITS.get(target, "")
            self.prediction_buttons[target].configure(text=f"{target} [{unit}]: ANALYZING...", text_color="yellow")
            
        threading.Thread(target=self.run_prediction_thread, args=(smiles,), daemon=True).start()

    def run_prediction_thread(self, smiles):
        targets = ["FP", "AIT", "HCOM", "FLVL", "FLVU", "ESOL"]
        for target in targets:
            try:
                res = self.wrapper.predict(smiles, target)
                if res is not None:
                    self.update_prediction_ui(target, f"{res:6.3f}", "green")
                else:
                    self.update_prediction_ui(target, "ERROR", "red")
            except Exception as e:
                self.log(f"[!] Inference failed for {target}: {e}")
                self.update_prediction_ui(target, "FAILED", "red")
        
        try:
            from src.smiles2data import smiles2data
            data = smiles2data(smiles, 0)
            if data != -1:
                # Pre-process ALL data to NumPy here to avoid tensor operations in the render thread
                self.current_data = {
                    'anum': data.mol_x[:, 0].detach().cpu().numpy().astype(int),
                    'coord': data.position.astype(float),
                    'bonds': np.array(data.mol_edge_index).astype(int)
                }
                self.update_lab_plot()
        except Exception as e:
            self.log(f"[!] Lab visualization failed: {e}")

        self.log("[*] Analysis complete! Click a target to view X-Ray importance.")
        self.root.after(0, lambda: self.run_button.configure(state="normal"))

    def select_target(self, target):
        if not self.current_smiles:
            self.log("[!] ERROR: Run analysis first!")
            return
        
        self.current_target = target
        self.log(f"[*] Switching focus to: {target}")
        self.xray_label.configure(text=f"ZONE C: X-RAY [FOCUS: {target}]")
        
        for t, btn in self.prediction_buttons.items():
            if t == target:
                btn.configure(border_color="cyan", border_width=2)
            else:
                btn.configure(border_color="#444444", border_width=1)

        if target in self.explanation_cache:
            self.update_xray_plot(self.explanation_cache[target])
        else:
            self.log(f"[*] Calculating importance for {target}...")
            self.warning_overlay.place_forget()
            threading.Thread(target=self.run_explanation_thread, args=(self.current_smiles, target), daemon=True).start()

    def run_explanation_thread(self, smiles, target):
        try:
            data, scores = self.wrapper.explain(smiles, target)
            if scores is not None:
                scores_np = np.array(scores).flatten()
                self.explanation_cache[target] = scores_np
                self.update_xray_plot(scores_np)
            else:
                self.show_xray_warning(f"UNAVAILABLE: {target}")
        except Exception as e:
            self.log(f"[!] Explanation failed for {target}: {e}")
            self.show_xray_warning(f"ERROR: {target}")

    def show_xray_warning(self, msg):
        def _show():
            self.warning_label.configure(text=f"[!] {msg}\nCheck model weights/logs")
            self.warning_overlay.place(relx=0.5, rely=0.5, relwidth=0.8, relheight=0.4, anchor="center")
        self.root.after(0, _show)

    def update_lab_plot(self):
        def _update():
            if self._drawing: return
            self._drawing = True
            try:
                self.ax_lab.clear()
                self.ax_lab.set_facecolor('#2b2b2b')
                self._plot_molecule_numpy(self.ax_lab, None)
                self.canvas_lab.draw_idle()
            except Exception as e:
                self.log(f"[!] Render error in Lab: {e}")
            finally:
                self._drawing = False
        self.root.after(0, _update)

    def update_xray_plot(self, scores_np):
        def _update():
            if self._drawing: return
            self._drawing = True
            try:
                self.warning_overlay.place_forget()
                self.ax_xray.clear()
                self.ax_xray.set_facecolor('#2b2b2b')
                p = self._plot_molecule_numpy(self.ax_xray, scores_np)
                
                if p:
                    self.cax_xray.clear()
                    self.cax_xray.set_axis_on()
                    cbar = self.fig_xray.colorbar(p, cax=self.cax_xray)
                    cbar.set_label('Importance', color='white', fontname="Orbitron", fontsize=10)
                    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
                
                # Immediate POV Sync
                self.ax_xray.view_init(elev=self.ax_lab.elev, azim=self.ax_lab.azim)
                self.ax_xray.set_xlim(self.ax_lab.get_xlim())
                self.ax_xray.set_ylim(self.ax_lab.get_ylim())
                self.ax_xray.set_zlim(self.ax_lab.get_zlim())
                self.canvas_xray.draw_idle()
            except Exception as e:
                self.log(f"[!] Render error in X-Ray: {e}")
            finally:
                self._drawing = False
        self.root.after(0, _update)

    def on_sync_view(self, event):
        now = time.time()
        if now - self._last_sync_time < 0.05: return # Throttle sync updates
        self._last_sync_time = now

        if event.inaxes == self.ax_lab:
            self.ax_xray.view_init(elev=self.ax_lab.elev, azim=self.ax_lab.azim)
            self.ax_xray.set_xlim(self.ax_lab.get_xlim())
            self.ax_xray.set_ylim(self.ax_lab.get_ylim())
            self.ax_xray.set_zlim(self.ax_lab.get_zlim())
            self.canvas_xray.draw_idle()
        elif event.inaxes == self.ax_xray:
            self.ax_lab.view_init(elev=self.ax_xray.elev, azim=self.ax_xray.azim)
            self.ax_lab.set_xlim(self.ax_xray.get_xlim())
            self.ax_lab.set_ylim(self.ax_xray.get_ylim())
            self.ax_lab.set_zlim(self.ax_xray.get_zlim())
            self.canvas_lab.draw_idle()

    def _plot_molecule_numpy(self, ax, scores):
        if not self.current_data: return None
        p = None
        try:
            anum = self.current_data['anum']
            coord = self.current_data['coord']
            bonds = self.current_data['bonds']
            
            if scores is not None:
                p = ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], s=200, c=scores, cmap="magma", alpha=0.8)
            else:
                ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], s=100, color='#00aaff', alpha=0.6)

            for c in range(coord.shape[0]):
                ax.text(coord[c, 0], coord[c, 1], coord[c, 2], SYMBOL.get(int(anum[c]), "?"),
                        ha='center', va='center', color='white', fontsize=10)

            for i, j in bonds:
                ax.plot((coord[i, 0], coord[j, 0]),
                        (coord[i, 1], coord[j, 1]),
                        (coord[i, 2], coord[j, 2]),
                        color='gray', linewidth=1, alpha=0.5)
            
            ax.set_axis_off()
        except Exception as e:
            print(f"Internal Plot Error: {e}")
        return p

    def update_prediction_ui(self, target, value, color):
        unit = self.UNITS.get(target, "")
        if not self.wrapper.weight_status.get(target, False):
            display_text = f"{target} [{unit}]: {value} (*)"
            display_color = "orange"
        else:
            display_text = f"{target} [{unit}]: {value}"
            display_color = color
        self.root.after(0, lambda: self.prediction_buttons[target].configure(text=display_text, text_color=display_color))

    def setup_tk_layout(self):
        pass

if __name__ == "__main__":
    if ctk:
        app = ctk.CTk()
        gui = ChemomileGUI(app)
        app.mainloop()
    else:
        app = tk.Tk()
        gui = ChemomileGUI(app)
        app.mainloop()

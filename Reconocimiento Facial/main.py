"""Interfaz gráfica del Sistema de Reconocimiento Facial."""
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

BASE = Path(__file__).resolve().parent
CODE = BASE / "code"
sys.path.insert(0, str(CODE))
import config  # noqa: E402  (rutas y utilidades compartidas)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc = None
        self.lines = queue.Queue()
        self.buttons = []

        root.title("Sistema de Reconocimiento Facial")
        root.geometry("1100x700")
        root.minsize(800, 550)
        root.configure(bg="#F1F5F9")
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=3)
        root.grid_rowconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.build_header()
        self.build_panel()
        self.build_console()
        self.refresh_people()
        self.refresh_status()
        self.show("Sistema iniciado correctamente.\n"
                  "Seleccione una operación en el panel izquierdo.")

    # ------------------------------------------------------------------ UI
    def build_header(self):
        header = tk.Frame(self.root, bg="#172B4D", height=75)
        header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        header.grid_propagate(False)
        tk.Label(header, text="SISTEMA DE RECONOCIMIENTO FACIAL",
                 font=("Segoe UI", 18, "bold"), bg="#172B4D", fg="white"
                 ).pack(side="left", padx=25, pady=20)

    def build_panel(self):
        panel = tk.Frame(self.root, bg="#E2E8F0", padx=20, pady=20)
        panel.grid(row=1, column=0, sticky="nsew")
        self.panel = panel

        tk.Label(panel, text="PANEL DE CONTROL", font=("Segoe UI", 13, "bold"),
                 bg="#E2E8F0", fg="#172B4D").pack(pady=(5, 20))
        tk.Label(panel, text="Nombre de persona:", font=("Segoe UI", 10),
                 bg="#E2E8F0").pack(anchor="w")

        # Editable: se puede elegir una persona existente o escribir una nueva.
        self.name_var = tk.StringVar()
        self.combo = ttk.Combobox(panel, textvariable=self.name_var,
                                  font=("Segoe UI", 12))
        self.combo.pack(fill="x", ipady=6, pady=(5, 20))

        self.add_button("Captura de cara", self.capture)
        self.add_button("Reconocimiento facial", self.recognize)
        self.add_button("Entrenar modelo", self.train)
        self.add_button("Borrar persona", self.delete_person, "#DC2626")

        self.stop_btn = tk.Button(
            panel, text="Detener proceso", command=self.stop, state="disabled",
            font=("Segoe UI", 10, "bold"), bg="#475569", fg="white",
            activebackground="#334155", activeforeground="white",
            relief="flat", cursor="hand2", pady=12)
        self.stop_btn.pack(fill="x", pady=(20, 7))

        self.status = tk.Label(panel, text="", font=("Segoe UI", 9),
                               bg="#E2E8F0", fg="#334155", justify="left",
                               wraplength=240)
        self.status.pack(side="bottom", anchor="w")

    def add_button(self, text, command, color="#2563EB"):
        btn = tk.Button(self.panel, text=text, command=command,
                        font=("Segoe UI", 10, "bold"), bg=color, fg="white",
                        activebackground="#1D4ED8", activeforeground="white",
                        relief="flat", cursor="hand2", pady=12)
        btn.pack(fill="x", pady=7)
        self.buttons.append(btn)

    def build_console(self):
        frame = tk.Frame(self.root, bg="#F8FAFC", padx=20, pady=20)
        frame.grid(row=1, column=1, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        tk.Label(frame, text="CONSOLA DE RESULTADOS", font=("Segoe UI", 14, "bold"),
                 bg="#F8FAFC", fg="#172B4D").grid(row=0, column=0, sticky="w",
                                                  pady=(0, 15))
        self.console = scrolledtext.ScrolledText(
            frame, font=("Consolas", 10), bg="#FFFFFF", fg="#334155",
            insertbackground="#172B4D", relief="flat", wrap=tk.WORD,
            padx=12, pady=12)
        self.console.grid(row=1, column=0, sticky="nsew")

    # ------------------------------------------------------------- helpers
    def show(self, text: str):
        self.console.delete("1.0", tk.END)
        self.console.insert(tk.END, text)

    def append(self, text: str):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    def refresh_people(self):
        self.combo["values"] = config.list_people()

    def refresh_status(self):
        if config.MODEL_PATH.exists():
            meta = config.load_meta()
            n = len(meta["people"])
            self.status.config(text=f"Modelo: {meta['algorithm']} · "
                                    f"{n} persona(s) entrenada(s)")
        else:
            self.status.config(text="Modelo: sin entrenar")

    def get_name(self):
        """Nombre válido del campo, o None (mostrando el motivo)."""
        raw = self.name_var.get()
        name = config.safe_name(raw)
        if not name:
            self.show("Ingrese un nombre válido (letras, números, espacios, "
                      "'-' o '_', máximo 50 caracteres).")
        return name

    # ------------------------------------------------------------ procesos
    def run_script(self, script: str, args=None, on_done=None):
        if self.proc is not None:
            messagebox.showinfo("En ejecución", "Ya hay un proceso en marcha.")
            return
        path = CODE / script
        if not path.exists():
            self.show(f"No se encontró el archivo:\n{path}")
            return

        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", str(path), *(args or [])],
                cwd=str(CODE), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1)
        except OSError as e:
            self.proc = None
            self.show(f"Error al iniciar el proceso: {e}")
            return

        self.show(f"Ejecutando {script}...\n\n")
        self.set_busy(True)
        threading.Thread(target=self.read_output, args=(self.proc,),
                         daemon=True).start()
        self.root.after(100, self.poll, on_done)

    def read_output(self, proc):
        for line in proc.stdout:
            self.lines.put(line)
        proc.wait()
        self.lines.put(None)  # fin

    def poll(self, on_done):
        finished = False
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    finished = True
                    break
                self.append(line)
        except queue.Empty:
            pass

        if finished:
            code = self.proc.returncode if self.proc else 0
            self.proc = None
            self.set_busy(False)
            self.append("\n--- Proceso finalizado ---" if code == 0
                        else f"\n--- Proceso terminó con código {code} ---")
            self.refresh_people()
            self.refresh_status()
            if on_done:
                on_done()
        else:
            self.root.after(100, self.poll, on_done)

    def set_busy(self, busy: bool):
        for b in self.buttons:
            b.config(state="disabled" if busy else "normal")
        self.stop_btn.config(state="normal" if busy else "disabled")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()

    def on_close(self):
        self.stop()
        self.root.destroy()

    # ------------------------------------------------------------ acciones
    def capture(self):
        name = self.get_name()
        if name:
            self.run_script("Faces.py", [name])

    def recognize(self):
        self.run_script("FacialRecognizer.py")

    def train(self):
        self.run_script("ModelsFaceRecognizer.py")

    def delete_person(self):
        name = self.get_name()
        if not name:
            return
        folder = config.DATA_DIR / name
        if not folder.is_dir():
            self.show(f"La persona '{name}' no existe.")
            return
        n = len(config.list_images(folder))
        if not messagebox.askyesno(
                "Confirmar eliminación",
                f"¿Eliminar a '{name}' y sus {n} imágenes?\n"
                "Después deberá volver a entrenar el modelo."):
            return
        try:
            shutil.rmtree(folder)
        except OSError as e:
            self.show(f"Error: {e}")
            return
        self.name_var.set("")
        self.refresh_people()
        self.show(f"'{name}' eliminada correctamente.\n"
                  "Recuerde pulsar «Entrenar modelo».")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()

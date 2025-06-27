# consciousness_experiment/agent_ui.py
# Standalone UI for OSCAR-C Cognitive Agent (UI Responsiveness Focus)

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
import queue
import threading
import datetime
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any # Added Dict, Any
import sys

logger_agent_ui = logging.getLogger("AgentUI_Standalone")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)-7s] [%(name)-30s] %(message)s")

LOGIC_AVAILABLE_UI = False; AgentController = None; AgentState = None
if __name__ == '__main__':
    try:
        current_script_path = Path(__file__).resolve(); agent_package_root = current_script_path.parent
        package_container_dir = str(agent_package_root.parent)
        if package_container_dir not in sys.path:
            sys.path.insert(0, package_container_dir)
        from consciousness_experiment.agent_controller import AgentController
        from consciousness_experiment.agent_state import AgentState
        LOGIC_AVAILABLE_UI = True
        logger_agent_ui.info("__main__: Successfully imported AgentController and AgentState.")
    except ImportError as e_abs: logger_agent_ui.critical(f"__main__: CRITICAL - Absolute import failed: {e_abs}.")
    except Exception as e_path_setup: logger_agent_ui.critical(f"__main__: CRITICAL - Error in path setup/import: {e_path_setup}")
else:
    try:
        from ..agent_controller import AgentController
        from ..agent_state import AgentState
        LOGIC_AVAILABLE_UI = True
        logger_agent_ui.info("Module Mode: Successfully imported AgentController and AgentState.")
    except ImportError as e_rel: logger_agent_ui.error(f"Module Mode: CRITICAL - Relative import failed: {e_rel}.")

if not LOGIC_AVAILABLE_UI:
    class AgentState: # type: ignore
        STOPPED="F_STOP"; STARTING="F_START"; RUNNING="F_RUN"; STOPPING="F_STOPP"; ERROR="F_ERR"; PAUSED="F_PAUSE"
    class AgentController: # type: ignore
        def __init__(self,q,m,c):pass;
        def start(self):pass;
        def stop(self):pass;
        def handle_user_input(self,t):pass
        def pause_agent(self): pass
        def resume_agent(self): pass
        def set_auto_pause(self, cycles: Optional[int]): pass
    logger_agent_ui.error("Using fallback AgentController/AgentState.")

class StandaloneExtensionAPI:
    def __init__(self): self.root: Optional[tk.Tk] = None
    def set_status(self, m: str, c: str="#333"): logger_agent_ui.info(f"API_STATUS: {m} ({c})")
    def get_selected_ollama_model(self) -> Optional[str]: return None

try:
    import ollama
    OLLAMA_AVAILABLE_UI = True
except ImportError:
    ollama = None; OLLAMA_AVAILABLE_UI = False; logger_agent_ui.error("ollama library not found.")

QUEUE_CHECK_INTERVAL_MS = 100 # Check queue more often, but process less each time
MAX_QUEUE_ITEMS_PER_TICK = 5  # Process fewer items per tick to yield to UI

class AgentUI:
    def __init__(self, root_window: tk.Tk, api: StandaloneExtensionAPI):
        logger_agent_ui.debug("Initializing AgentUI instance...")
        self.root = root_window
        self.api = api
        self.api.root = root_window
        self.agent_controller_instance: Optional[AgentController] = None
        self.agent_state_internal = AgentState.STOPPED if LOGIC_AVAILABLE_UI else "FALLBACK_STOPPED" # Internal state tracker
        self.model_list: list[str] = []
        self.ui_queue = queue.Queue()

        self.cs_level_var = tk.StringVar(value="CS: UNKNOWN")
        self.workspace_load_var = tk.StringVar(value="WS: 0/0")
        self.drives_var = tk.StringVar(value="Drives: C:0.0 S:0.0 E:0.0")
        self.active_goal_var = tk.StringVar(value="Goal: None")
        self.status_label_var = tk.StringVar(value="Status: STOPPED") # For overall agent status

        self.php_pain_var = tk.StringVar(value="Pain: 0.0")
        self.php_happiness_var = tk.StringVar(value="Happy: 5.0")
        self.php_purpose_var = tk.StringVar(value="Purp: 5.0")
        self.php_age_var = tk.StringVar(value="Age: 0")
        self.php_pain_sources_var = tk.StringVar(value="PainSrcs: 0")
        self.php_top_pain_desc_var = tk.StringVar(value="TopPain: None")


        if not LOGIC_AVAILABLE_UI:
            ttk.Label(self.root, text="CRITICAL ERROR:\nAgent logic modules failed to load.",
                      foreground="red", wraplength=450, font=('Segoe UI', 12, 'bold'),
                      justify=tk.CENTER).pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
            return
        self._setup_widgets()
        self.root.after(50, self._update_ui_from_internal_state) # Initial UI state update
        if OLLAMA_AVAILABLE_UI: self.root.after(100, self._fetch_models)
        else: self.root.after(100, lambda: self._update_models_dropdown([], "Ollama library not found."))
        self.root.after(QUEUE_CHECK_INTERVAL_MS, self._periodic_queue_check) # Start queue checker
        logger_agent_ui.info("AgentUI initialized successfully.")

    def _setup_widgets(self):
        main_content_frame = ttk.Frame(self.root, padding=10); main_content_frame.pack(fill=tk.BOTH, expand=True)
        self.control_frame = ttk.Frame(main_content_frame); self.control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5))
        self.start_button = ttk.Button(self.control_frame, text="▶ Start Agent", command=self._start_agent, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=(0,5))
        self.stop_button = ttk.Button(self.control_frame, text="■ Stop Agent", command=self._stop_agent, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0,5))
        
        self.pause_resume_button = ttk.Button(self.control_frame, text="❚❚ Pause", command=self._toggle_pause_agent, state=tk.DISABLED)
        self.pause_resume_button.pack(side=tk.LEFT, padx=(0,10))

        ttk.Label(self.control_frame, text="Model:").pack(side=tk.LEFT, padx=(0,2))
        self.model_var = tk.StringVar(); self.model_combo = ttk.Combobox(self.control_frame, textvariable=self.model_var, width=30, state=tk.DISABLED)
        self.model_combo.pack(side=tk.LEFT, padx=(0,5))
        self.refresh_models_button = ttk.Button(self.control_frame, text="\U0001F504", width=3, command=self._fetch_models, state=tk.DISABLED)
        self.refresh_models_button.pack(side=tk.LEFT, padx=(0,10))

        ttk.Label(self.control_frame, text="AutoPause:").pack(side=tk.LEFT, padx=(5,0))
        self.auto_pause_cycles_var = tk.StringVar(value="0") # Default 0 or empty
        self.auto_pause_entry = ttk.Entry(self.control_frame, textvariable=self.auto_pause_cycles_var, width=4)
        self.auto_pause_entry.pack(side=tk.LEFT, padx=(0,2))
        self.set_auto_pause_button = ttk.Button(self.control_frame, text="Set", command=self._set_auto_pause_from_ui, state=tk.DISABLED, width=4)
        self.set_auto_pause_button.pack(side=tk.LEFT, padx=(0,10))

        self.status_display_label_widget = ttk.Label(self.control_frame, textvariable=self.status_label_var, foreground="gray", anchor=tk.E)
        self.status_display_label_widget.pack(side=tk.RIGHT, padx=(5,0), fill=tk.X, expand=True)

        status_meters_frame = ttk.Frame(main_content_frame); status_meters_frame.pack(side=tk.TOP, fill=tk.X, pady=(5,5))
        ttk.Label(status_meters_frame, textvariable=self.cs_level_var, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Label(status_meters_frame, textvariable=self.workspace_load_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(status_meters_frame, textvariable=self.drives_var, width=25).pack(side=tk.LEFT, padx=2)
        active_goal_label = ttk.Label(status_meters_frame, textvariable=self.active_goal_var, anchor=tk.W)
        active_goal_label.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- New Status Meters Row for P/H/P ---
        php_status_meters_frame = ttk.Frame(main_content_frame)
        php_status_meters_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5)) # Place it below the first status row

        ttk.Label(php_status_meters_frame, textvariable=self.php_pain_var, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(php_status_meters_frame, textvariable=self.php_happiness_var, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(php_status_meters_frame, textvariable=self.php_purpose_var, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(php_status_meters_frame, textvariable=self.php_age_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(php_status_meters_frame, textvariable=self.php_pain_sources_var, width=12).pack(side=tk.LEFT, padx=2)
        top_pain_label = ttk.Label(php_status_meters_frame, textvariable=self.php_top_pain_desc_var, anchor=tk.W)
        top_pain_label.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)


        output_frame = ttk.LabelFrame(main_content_frame, text="Agent Output / Log")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=5)
        self.output_display = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=('Consolas',9))
        self.output_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        try:
            for tag, fg, *font_style in [("user_tag", "#007ACC", 'bold'), ("agent_tag", "#228822"),
                                         ("system_tag", "#666666", 'italic'), ("error_tag", "red", 'bold'),
                                         ("warn_tag", "#FFA500"), ("debug_tag", "#0000CD"),
                                         ("agent_log_error", "darkred"),("agent_log_warn", "darkorange"),
                                         ("agent_log_info", "darkgreen"), ("agent_log_debug", "darkblue")]:
                font_options = ('Consolas', 9, *font_style)
                self.output_display.tag_configure(tag, foreground=fg, font=font_options, lstrip=True if "agent_log" in tag else False, rstrip=True if "agent_log" in tag else False)
        except tk.TclError as e: logger_agent_ui.warning(f"Tag config error: {e}")

        interaction_frame = ttk.Frame(main_content_frame); interaction_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0), padx=0)
        self.input_entry = ttk.Entry(interaction_frame, font=('Segoe UI',10), state=tk.DISABLED)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.input_entry.bind("<Return>", lambda event=None: self._send_input_event(event))
        self.send_button = ttk.Button(interaction_frame, text="➤ Send", command=self._send_input, state=tk.DISABLED)
        self.send_button.pack(side=tk.RIGHT)

    def _log_message(self, level: str, message: str, to_console_too: bool = True):
        # ... (Same as previous) ...
        if to_console_too: getattr(logger_agent_ui, level.lower(), logger_agent_ui.info)(message)
        if not (hasattr(self, 'output_display') and self.output_display.winfo_exists()): return
        try:
            level_upper = level.upper(); tag = {"INFO":"system_tag","ERROR":"error_tag","WARN":"warn_tag","DEBUG":"debug_tag"}.get(level_upper,"system_tag")
            prefix = f"[{level_upper}]" if level_upper != "INFO" else "[UI System]"
            self.output_display.config(state=tk.NORMAL)
            self.output_display.insert(tk.END, f"{datetime.datetime.now().strftime('%H:%M:%S')} {prefix} {message}\n", tag)
            self.output_display.config(state=tk.DISABLED); self.output_display.yview(tk.END)
        except Exception as e: logger_agent_ui.error(f"UI log error: {e}", exc_info=False)


    def _fetch_models(self):
        # ... (Logic to disable buttons and start thread same as before, using self.agent_state_internal) ...
        if not OLLAMA_AVAILABLE_UI: self._log_message("ERROR","Ollama NA"); self._update_ui_from_internal_state(); return
        if LOGIC_AVAILABLE_UI and self.agent_state_internal != AgentState.STOPPED:
            self._log_message("WARN","Agent active, stop before model refresh.");
            if self.root: messagebox.showwarning("Agent Active","Stop agent first.",parent=self.root)
            self._update_ui_from_internal_state(); return
        self._log_message("INFO","Fetching models…")
        try:
            self.model_var.set("[Fetching…]"); self.model_combo.config(state=tk.DISABLED,values=[])
            for widget_name in ["refresh_models_button", "start_button"]:
                widget = getattr(self, widget_name, None)
                if widget and widget.winfo_exists(): widget.config(state=tk.DISABLED)
        except tk.TclError: pass
        threading.Thread(target=self._model_fetch_thread, daemon=True).start()

    def _model_fetch_thread(self):
        # ... (Same robust parsing as before) ...
        models, error_msg, processed_model_names = [], None, []
        try:
            listed = ollama.list(); models_data_list = listed.get('models', [])
            if isinstance(models_data_list, list):
                for item_data in models_data_list:
                    model_id=None
                    if isinstance(item_data,dict): model_id=item_data.get('model',item_data.get('name'))
                    elif hasattr(item_data,'model'): model_id=getattr(item_data,'model',None)
                    elif hasattr(item_data,'name'): model_id=getattr(item_data,'name',None)
                    if model_id and isinstance(model_id,str): processed_model_names.append(model_id)
                models=sorted(list(set(processed_model_names)))
            else: error_msg="Ollama: 'models' not a list"
        except Exception as e: error_msg=str(e); logger_agent_ui.exception("Ollama fetch error:")
        if self.root.winfo_exists(): self.root.after(0, lambda: self._handle_model_fetch_result(models, error_msg))


    def _handle_model_fetch_result(self, models: List[str], error: Optional[str]):
        """Handles the result from the model fetch thread in the UI thread."""
        self._update_models_dropdown(models, error)
        self._update_ui_from_internal_state() # Re-enable buttons based on result

    def _update_models_dropdown(self, models: list[str], error: Optional[str]):
        # ... (Same as before, using self._log_message) ...
        if not (hasattr(self, 'model_combo') and self.model_combo.winfo_exists()): return
        self.model_list = models
        try:
            if error: self.model_var.set("[Error]"); self.model_combo['values']=[]; self._log_message("ERROR",f"Model fetch: {error}")
            elif models:
                self.model_combo['values']=models; current_val=self.model_var.get()
                preferred=next((m for m in models if "llama3" in m.lower()), models[0] if models else "")
                if current_val in models and not current_val.startswith("["): self.model_var.set(current_val)
                elif preferred and preferred in models: self.model_var.set(preferred)
                elif models: self.model_var.set(models[0])
                else: self.model_var.set("[No Models]")
                self._log_message("INFO", f"Models loaded ({len(models)}). Sel: {self.model_var.get()}")
            else: self.model_var.set("[No Models]"); self.model_combo['values']=[]; self._log_message("WARN","Ollama reports 0 models.")
        except tk.TclError: pass # Ignore if widget destroyed

    def _update_ui_from_internal_state(self):
        """Centralized method to update UI widget states based on self.agent_state_internal."""
        if not LOGIC_AVAILABLE_UI: return
        current = self.agent_state_internal
        is_stopped = (current == AgentState.STOPPED)
        is_running = (current == AgentState.RUNNING)
        is_paused = (current == AgentState.PAUSED)
        models_loaded = bool(self.model_list)

        can_start = is_stopped and models_loaded and OLLAMA_AVAILABLE_UI
        can_stop = is_running or current == AgentState.STARTING or current == AgentState.ERROR or is_paused # Allow stop when paused
        can_interact = is_running # Keep interaction disabled when paused
        can_fetch = is_stopped and OLLAMA_AVAILABLE_UI
        can_select_model = is_stopped and models_loaded

        can_pause_resume = is_running or is_paused
        can_set_auto_pause = is_running or is_paused # Allow setting auto-pause even if already paused (will apply on next resume)

        try:
            # Update existing widget states
            for widget_name, can_enable in [("start_button", can_start), ("stop_button", can_stop),
                                            ("send_button", can_interact), ("input_entry", can_interact),
                                            ("refresh_models_button", can_fetch)]:
                widget = getattr(self, widget_name, None)
                if widget and widget.winfo_exists(): widget.config(state=tk.NORMAL if can_enable else tk.DISABLED)
            
            # New Pause/Resume Button
            pause_resume_widget = getattr(self, 'pause_resume_button', None)
            if pause_resume_widget and pause_resume_widget.winfo_exists():
                pause_resume_widget.config(state=tk.NORMAL if can_pause_resume else tk.DISABLED)
                if is_paused:
                    pause_resume_widget.config(text="▶ Resume")
                else: # RUNNING or other states
                    pause_resume_widget.config(text="❚❚ Pause")
            
            # New Auto-Pause Widgets
            auto_pause_entry_widget = getattr(self, 'auto_pause_entry', None)
            set_auto_pause_button_widget = getattr(self, 'set_auto_pause_button', None)
            if auto_pause_entry_widget and auto_pause_entry_widget.winfo_exists():
                auto_pause_entry_widget.config(state=tk.NORMAL if can_set_auto_pause else tk.DISABLED)
            if set_auto_pause_button_widget and set_auto_pause_button_widget.winfo_exists():
                set_auto_pause_button_widget.config(state=tk.NORMAL if can_set_auto_pause else tk.DISABLED)

            model_combo_widget = getattr(self, 'model_combo', None)
            if model_combo_widget and model_combo_widget.winfo_exists():
                 model_combo_widget.config(state="readonly" if can_select_model else tk.DISABLED)


            status_label_var_widget = getattr(self, 'status_label_var', None)
            status_display_label_widget = getattr(self, 'status_display_label_widget', None)
            if status_label_var_widget and status_display_label_widget and status_display_label_widget.winfo_exists():
                status_text = f"Status: {current.name if hasattr(current, 'name') else str(current)}"
                color_map = { 
                    AgentState.RUNNING: "green", AgentState.ERROR: "red", 
                    AgentState.STARTING: "blue", AgentState.STOPPING: "orange", 
                    AgentState.STOPPED: "gray",
                    AgentState.PAUSED: "purple" # New color for PAUSED
                }
                color = color_map.get(current, "black")
                status_label_var_widget.set(status_text)
                status_display_label_widget.config(foreground=color)
        except tk.TclError: pass # Ignore if widgets are being destroyed

    def _toggle_pause_agent(self):
        if not self.agent_controller_instance:
            self._log_message("WARN", "Toggle Pause: Agent controller not available.")
            return
        
        if self.agent_state_internal == AgentState.RUNNING:
            self._log_message("INFO", "UI Requesting Agent PAUSE...")
            self.agent_controller_instance.pause_agent()
        elif self.agent_state_internal == AgentState.PAUSED:
            self._log_message("INFO", "UI Requesting Agent RESUME...")
            self.agent_controller_instance.resume_agent()
        else:
            self._log_message("WARN", f"Toggle Pause: Agent not in RUNNING or PAUSED state (is {self.agent_state_internal.name if hasattr(self.agent_state_internal, 'name') else 'Unknown'}).")

    def _set_auto_pause_from_ui(self):
        if not self.agent_controller_instance:
            self._log_message("WARN", "Set Auto-Pause: Agent controller not available.")
            return
        
        try:
            cycles_str = self.auto_pause_cycles_var.get()
            if not cycles_str.strip(): # If empty, disable auto-pause
                num_cycles = 0
            else:
                num_cycles = int(cycles_str)

            if num_cycles <= 0:
                self._log_message("INFO", "UI Requesting: Disable auto-pause.")
                self.agent_controller_instance.set_auto_pause(None)
            else:
                self._log_message("INFO", f"UI Requesting: Set auto-pause for {num_cycles} cycles.")
                self.agent_controller_instance.set_auto_pause(num_cycles)
        except ValueError:
            self._log_message("ERROR", "Invalid number of cycles for auto-pause. Please enter an integer.")
            if self.root: messagebox.showerror("Input Error", "Auto-pause cycles must be a whole number.", parent=self.root)
        except Exception as e:
            self._log_message("ERROR", f"Error setting auto-pause: {e}")

    def _start_agent(self):
        if not LOGIC_AVAILABLE_UI: messagebox.showerror("Error","Agent logic missing.",parent=self.root); return
        if self.agent_state_internal != AgentState.STOPPED: return
        selected_model = self.model_var.get()
        if not selected_model or selected_model.startswith("[") or selected_model not in self.model_list:
             messagebox.showerror("Error", "Select a valid model.", parent=self.root); return
        self._log_message("INFO", f"Starting agent (Model: {selected_model})...")
        
        self.agent_state_internal = AgentState.STARTING
        self._update_ui_from_internal_state() # Update UI immediately

        # Start agent controller in a new thread to avoid blocking UI
        def agent_thread_target():
            try:
                script_dir = Path(__file__).parent.resolve()
                config_path = script_dir / "config.toml"
                config_path_str = str(config_path) if config_path.exists() else "config.toml"
                if not config_path.exists(): self.ui_queue.put(("log_warn", f"config.toml not found at {config_path}."))
                
                self.agent_controller_instance = AgentController(self.ui_queue, model_name=selected_model, config_path=config_path_str)
                self.agent_controller_instance.start() # This blocks until agent's asyncio loop finishes
                # When start() returns, agent is stopped or errored.
                logger_agent_ui.info("AgentController has finished its run.")
            except Exception as e_thread:
                logger_agent_ui.exception("Exception in agent thread:")
                self.ui_queue.put(("log_error", f"Agent Thread Error: {e_thread}"))
                self.ui_queue.put(("state_update", AgentState.ERROR))
            finally:
                # Ensure state is updated if thread exits unexpectedly
                if self.agent_state_internal not in [AgentState.STOPPED, AgentState.ERROR]:
                    self.ui_queue.put(("state_update", AgentState.STOPPED))
        
        threading.Thread(target=agent_thread_target, daemon=True).start()

    def _stop_agent(self):
        if not LOGIC_AVAILABLE_UI: return
        allowed_states = [AgentState.RUNNING, AgentState.STARTING, AgentState.ERROR, AgentState.PAUSED]
        if not self.agent_controller_instance or self.agent_state_internal not in allowed_states:
            if not self.agent_controller_instance and self.agent_state_internal != AgentState.STOPPED:
                self.agent_state_internal = AgentState.STOPPED; self._update_ui_from_internal_state()
            return
        self._log_message("INFO", "Requesting agent stop...")
        self.agent_state_internal = AgentState.STOPPING; self._update_ui_from_internal_state()
        try:
            if self.agent_controller_instance: self.agent_controller_instance.stop()
        except Exception as e:
            self._log_message("ERROR", f"Stop Request Error: {e}")
            self.agent_state_internal = AgentState.ERROR; self._update_ui_from_internal_state()

    def _send_input_event(self, event=None): self._send_input(); return "break"
    def _send_input(self):
        if not LOGIC_AVAILABLE_UI or self.agent_state_internal != AgentState.RUNNING: return
        user_text = self.input_entry.get().strip()
        if not user_text: return
        self._append_output(f"You: {user_text}", "user_tag"); self.input_entry.delete(0, tk.END)
        try:
            if self.agent_controller_instance: self.agent_controller_instance.handle_user_input(user_text)
            else: self._log_message("ERROR","Agent controller NA."); self.agent_state_internal=AgentState.ERROR;self._update_ui_from_internal_state()
        except Exception as e: self._log_message("ERROR",f"Send Input Error: {e}")

    def _append_output(self, text: str, tag: str):
        if not (hasattr(self, 'output_display') and self.output_display.winfo_exists()): return
        try:
            self.output_display.config(state=tk.NORMAL)
            self.output_display.insert(tk.END, text + "\n\n", tag)
            self.output_display.config(state=tk.DISABLED); self.output_display.yview(tk.END)
        except tk.TclError: pass

    def _periodic_queue_check(self):
        if not self.root.winfo_exists(): return # Stop if root window is gone
        processed_count = 0
        try:
            while not self.ui_queue.empty() and processed_count < MAX_QUEUE_ITEMS_PER_TICK:
                processed_count += 1
                message_type, data = self.ui_queue.get_nowait()
                
                if message_type == "agent_output": self._append_output(f"{data}", "agent_tag")
                elif message_type.startswith("log_"):
                    level_str = message_type.split("_",1)[1]; tag=f"agent_log_{level_str.lower()}"
                    self._append_output(f"Agent: [{level_str.upper()}] {str(data)}", tag)
                elif message_type == "state_update":
                    if LOGIC_AVAILABLE_UI and isinstance(data, AgentState):
                        if self.agent_state_internal != data:
                            logger_agent_ui.info(f"Agent state update via queue: {self.agent_state_internal.name if hasattr(self.agent_state_internal,'name') else '?'} -> {data.name}") # type: ignore
                            self.agent_state_internal = data
                            if self.agent_state_internal in [AgentState.STOPPED, AgentState.ERROR]:
                                self.agent_controller_instance = None # Clear instance when fully stopped/errored
                        # Always call _update_ui_from_internal_state to refresh button states etc.
                        self._update_ui_from_internal_state()
                elif message_type == "model_update":
                    if isinstance(data,tuple) and len(data)==2: self._handle_model_fetch_result(data[0],data[1]) # Uses after(0)
                elif message_type == "agent_cs_level_update":
                    if isinstance(data,str): self.cs_level_var.set(f"CS: {data}")
                elif message_type == "agent_workspace_update":
                    if isinstance(data,dict): self.workspace_load_var.set(f"WS: {data.get('load',0)}/{data.get('capacity',0)}")
                elif message_type == "agent_drives_update":
                    if isinstance(data,dict):
                        c,s,e = data.get('curiosity',0.0), data.get('satisfaction',0.0), data.get('competence',0.0)
                        self.drives_var.set(f"Drives: C:{c:.2f} S:{s:.2f} E:{e:.2f}")
                elif message_type == "agent_goal_update":
                    if isinstance(data,dict):
                        desc,status = str(data.get('description','None'))[:40], data.get('status','N/A')
                        self.active_goal_var.set(f"Goal: {desc}... ({status})")
                    elif data is None: self.active_goal_var.set("Goal: None")
                elif message_type == "agent_php_update":
                    if isinstance(data, dict):
                        self.php_pain_var.set(f"Pain: {data.get('pain_level', 0.0):.2f}")
                        self.php_happiness_var.set(f"Happy: {data.get('happiness_level', 5.0):.2f}")
                        self.php_purpose_var.set(f"Purp: {data.get('purpose_level', 5.0):.2f}")
                        self.php_age_var.set(f"Age: {data.get('agent_age_cycles', 0)}")
                        self.php_pain_sources_var.set(f"PainSrcs: {data.get('active_pain_sources_count', 0)}")
                        
                        top_pains_summary = data.get("top_pain_sources_summary", [])
                        if top_pains_summary and isinstance(top_pains_summary, list):
                            summary_texts = [f"{p.get('description','?')} ({p.get('intensity',0):.1f})" for p in top_pains_summary]
                            self.php_top_pain_desc_var.set(f"TopPain: {', '.join(summary_texts)}")
                        else:
                            self.php_top_pain_desc_var.set("TopPain: None")
                else: logger_agent_ui.warning(f"Unknown UI queue message: {message_type}")
                self.ui_queue.task_done()
        except queue.Empty: pass
        except Exception as e: logger_agent_ui.exception("Error processing UI queue:")
        
        if self.root.winfo_exists(): # Schedule next check only if root still exists
            self.root.after(QUEUE_CHECK_INTERVAL_MS, self._periodic_queue_check)

# --- Main execution block for standalone UI ---
if __name__ == '__main__':
    standalone_logger = logging.getLogger() # Get root logger
    standalone_logger.setLevel(logging.DEBUG)
    for handler in standalone_logger.handlers[:]: standalone_logger.removeHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)-7s] [%(name)-40s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    standalone_logger.addHandler(console_handler)
    logger_agent_ui.info("--- Running OSCAR-C AgentUI in Standalone Mode ---")

    if not LOGIC_AVAILABLE_UI:
        logger_agent_ui.critical("Standalone: Core agent logic modules failed. UI will be non-functional.")

    root = tk.Tk()
    root.title("OSCAR-C Agent UI (Standalone)")
    root.geometry("850x700")
    api_instance = StandaloneExtensionAPI()
    app_ui_instance = AgentUI(root, api_instance)

    def on_standalone_close():
        logger_agent_ui.info("Standalone UI closing initiated...")
        if app_ui_instance.agent_controller_instance and hasattr(app_ui_instance.agent_controller_instance, 'stop'):
            logger_agent_ui.info("Attempting to stop active agent controller...")
            try: app_ui_instance.agent_controller_instance.stop()
            except Exception as e: logger_agent_ui.error(f"Error signaling agent controller stop: {e}")
        
        # Give agent thread a moment to process stop if it's running in a thread
        # This is a bit of a guess; proper thread joining or async shutdown is better
        # root.after(500, root.destroy) # Delay destroy slightly
        logger_agent_ui.info("Destroying Tkinter root window.")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_standalone_close)
    try: root.mainloop()
    except KeyboardInterrupt: logger_agent_ui.info("KeyboardInterrupt. Closing UI."); on_standalone_close()
    finally: logger_agent_ui.info("--- Standalone AgentUI Application Closed ---"); logging.shutdown()
# --- END OF FILE agent_ui.py ---
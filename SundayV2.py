#!/usr/bin/env python3
import sys
import os
import ctypes
import json
import traceback
import platform
import psutil
import numpy as np
import subprocess
import time
import difflib
import re
import tempfile
import signal
import threading
import urllib.request
import requests
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, QGroupBox,
                             QTabWidget, QTreeWidget, QTreeWidgetItem, QCheckBox, QSpinBox,
                             QFileDialog, QMessageBox, QDialog, QComboBox, QProgressBar, 
                             QMenu, QAction, QToolButton, QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRegExp, QEventLoop, QTimer
from PyQt5.QtGui import (QFont, QPalette, QColor, QTextCursor, QSyntaxHighlighter, 
                         QTextCharFormat, QBrush, QIcon)

# ======================
# OLLAMA BACKEND SYSTEM
# ======================
class OllamaInitializationThread(QThread):
    initialization_complete = pyqtSignal(bool, str)
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
    def run(self):
        try:
            # Check if Ollama server is running
            try:
                response = requests.get("http://localhost:11434")
                if response.status_code != 200:
                    self.initialization_complete.emit(False, "Ollama server not running. Please start Ollama.")
                    return
            except Exception:
                self.initialization_complete.emit(False, "Ollama server not found. Please install and start Ollama.")
                return
                
            # Check if model is available
            try:
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    self.initialization_complete.emit(False, "Failed to get model list from Ollama")
                    return
                    
                models = response.json().get("models", [])
                model_found = any(model["name"] == self.model_name for model in models)
                
                if not model_found:
                    self.initialization_complete.emit(False, f"Model {self.model_name} not found. Pulling model...")
                    # Pull model in background
                    threading.Thread(target=self.pull_model, daemon=True).start()
                    return
                    
                self.initialization_complete.emit(True, "Ollama connection initialized successfully")
            except Exception as e:
                self.initialization_complete.emit(False, f"Error checking model: {str(e)}")
        except Exception as e:
            self.initialization_complete.emit(False, f"Ollama initialization failed: {str(e)}")

    def pull_model(self):
        """Pull the model from Ollama repository"""
        try:
            response = requests.post(
                "http://localhost:11434/api/pull",
                json={"name": self.model_name},
                stream=True
            )
            
            # Stream the pull progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("status") == "success":
                        self.initialization_complete.emit(True, f"Model {self.model_name} pulled successfully")
                        return
        except Exception as e:
            self.initialization_complete.emit(False, f"Error pulling model: {str(e)}")

class OllamaManager:
    def __init__(self):
        self.model_name = "deepseek-coder:6.7b"  # Updated to a supported model
        self.model_loaded = False
        self.initialization_thread = None
        self.generation_lock = threading.Lock()
        self.active_generation = None
        
    def initialize_model(self, callback):
        """Initialize the Ollama connection in a background thread"""
        self.initialization_thread = OllamaInitializationThread(self.model_name)
        self.initialization_thread.initialization_complete.connect(callback)
        self.initialization_thread.start()
    
    def generate_response(self, prompt, history, timeout=300, thinking_mode=False):
        """Generate response using Ollama API with timeout"""
        if not self.model_loaded:
            return "Error: Ollama not loaded. Please ensure Ollama is installed and running."
            
        try:
            with self.generation_lock:
                self.active_generation = True
                
                # Build messages from history
                messages = []
                for message in history:
                    messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
                
                # Add current prompt
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                # For thinking mode, add system prompt
                if thinking_mode:
                    thinking_prompt = (
                        "You are in Thinking Mode. Please reason step by step about the problem. "
                        "Consider different approaches, analyze potential solutions, and think through the implementation. "
                        "After thorough reasoning, generate the final Python code solution. "
                        "Format your response as:\n\n"
                        "### Thinking Process ###\n"
                        "[Your step-by-step reasoning here]\n\n"
                        "### Final Code ###\n"
                        "```python\n"
                        "[Your Python code here]\n"
                        "```"
                    )
                    messages.insert(0, {
                        "role": "system",
                        "content": thinking_prompt
                    })
                
                # Create request payload
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False
                }
                
                # Send request to Ollama API
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code != 200:
                    return f"Error: API request failed with status {response.status_code}"
                
                response_data = response.json()
                
                if "message" not in response_data or "content" not in response_data["message"]:
                    return "Error: Invalid response format from Ollama API"
                
                return response_data["message"]["content"].strip()
                
        except Exception as e:
            return f"Error in generation process: {str(e)}"
        finally:
            self.active_generation = None

# ======================
# MEMORY SYSTEM
# ======================
class MemoryManager:
    def __init__(self):
        self.memory_file = "autocoder_memory.json"
        self.memory = self.load_memory()
        
    def load_memory(self):
        """Load memory from persistent storage"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"facts": [], "conversations": []}
    
    def save_memory(self):
        """Save memory to persistent storage"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=4)
        except Exception as e:
            print(f"Error saving memory: {str(e)}")
    
    def remember_fact(self, fact, importance=5):
        """Store a new fact in memory with importance rating"""
        self.memory["facts"].append({
            "fact": fact,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "importance": importance
        })
        # Keep only the most important facts
        self.memory["facts"].sort(key=lambda x: x["importance"], reverse=True)
        self.memory["facts"] = self.memory["facts"][:100]  # Limit to 100 facts
        self.save_memory()
    
    def remember_conversation(self, role, content):
        """Store a conversation snippet"""
        self.memory["conversations"].append({
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        # Keep only recent conversations
        self.memory["conversations"] = self.memory["conversations"][-50:]  # Last 50 messages
        self.save_memory()
    
    def get_relevant_memory(self, query, max_facts=5):
        """Retrieve relevant facts based on query similarity"""
        # Simple implementation - could be enhanced with embeddings
        relevant = []
        for fact in self.memory["facts"]:
            if query.lower() in fact["fact"].lower():
                relevant.append(fact)
                if len(relevant) >= max_facts:
                    break
        return relevant

# ======================
# MAIN APPLICATION
# ======================
class OllamaGenerationThread(QThread):
    response_generated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)  # Progress percentage

    def __init__(self, manager, prompt, history, thinking_mode=False):
        super().__init__()
        self.manager = manager
        self.prompt = prompt
        self.history = history
        self.is_running = True
        self.thinking_mode = thinking_mode

    def run(self):
        try:
            # Start progress updates
            self.start_progress_updates()
            
            # Generate response
            response = self.manager.generate_response(self.prompt, self.history, thinking_mode=self.thinking_mode)
            
            if not self.is_running:
                return  # Thread was stopped
                
            if response is None:
                self.error_occurred.emit("No response generated")
            elif response.startswith("Error"):
                self.error_occurred.emit(response)
            else:
                self.response_generated.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.stop_progress_updates()

    def start_progress_updates(self):
        """Simulate progress updates"""
        self.progress_value = 0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(500)  # Update every 500ms

    def update_progress(self):
        if not self.is_running:
            self.progress_timer.stop()
            return
            
        self.progress_value = min(self.progress_value + 2, 95)
        self.progress_updated.emit(self.progress_value)
        
        # Stop when we reach 95% (actual completion will jump to 100%)
        if self.progress_value >= 95:
            self.progress_timer.stop()

    def stop_progress_updates(self):
        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()
        self.progress_updated.emit(100)  # Complete
        
    def stop(self):
        self.is_running = False
        if hasattr(self.manager, 'active_generation'):
            self.manager.active_generation = False  # Signal to stop generation
        self.stop_progress_updates()

class AutoCoderGUI(QMainWindow):
    _instance = None
    plugin_output = pyqtSignal(str)
    update_code_requested = pyqtSignal(str)
    
    @classmethod
    def get_instance(cls):
        return cls._instance
    
    def __init__(self):
        super().__init__()
        AutoCoderGUI._instance = self
        self.setWindowTitle("AutoCoder - Ollama Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize Ollama
        self.ollama = OllamaManager()
        self.device_status = "CPU (Ollama)"
        
        # Initialize memory system
        self.memory = MemoryManager()
        
        # Initialize state
        self.conversation_history = []
        self.code_versions = []
        self.current_code = ""
        self.execution_thread = None
        self.current_goal = ""
        self.iteration_count = 0
        self.max_iterations = 10
        self.last_execution_output = ""
        self.debug_mode = False
        self.automatic_execution = False
        self.plugins = []
        self.active_threads = []
        self.generation_start_time = 0
        self.generation_timer = QTimer()
        self.generation_timer.timeout.connect(self.update_generation_status)
        self.thinking_mode_active = False
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create header
        header_layout = QHBoxLayout()
        
        header_label = QLabel("AutoCoder - Self-Improving AI with Ollama")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(header_label, 4)
        
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Create tab widget for main content
        self.tab_widget = QTabWidget()
        
        # Automation Tab
        automation_tab = self.create_automation_tab()
        self.tab_widget.addTab(automation_tab, "Automation")
        
        # Version History Tab
        history_tab = self.create_history_tab()
        self.tab_widget.addTab(history_tab, "Version History")
        
        # Settings Tab
        settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(settings_tab, "Settings")
        
        main_layout.addWidget(self.tab_widget, 1)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.iteration_label = QLabel("Iteration: 0")
        self.goal_label = QLabel("Current Goal: None")
        self.device_label = QLabel(f"Device: {self.device_status}")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.addPermanentWidget(self.iteration_label)
        self.status_bar.addPermanentWidget(self.goal_label)
        self.status_bar.addPermanentWidget(self.device_label)
        
        # Connect signals
        self.plugin_output.connect(self.handle_plugin_output)
        self.update_code_requested.connect(self.update_code)
        
        # Initialize plugins
        self.setup_plugins()
        
        # Load settings
        self.load_settings()
        
        # Set default instructions if not set
        if not hasattr(self, 'instructions_edit') or not self.instructions_edit.toPlainText().strip():
            self.reset_instructions_to_default()
        
        # Add initial message
        self.add_to_conversation("System", "AutoCoder initialized with Ollama backend.")
        self.add_to_conversation("System", f"Using device: {self.device_status}")
        self.status_label.setText("Initializing Ollama...")
        self.add_to_conversation("System", "Initializing Ollama connection...")
        
        # Initialize Ollama in a background thread
        self.ollama.initialize_model(self.handle_ollama_initialization)
        
        # Show progress bar during initialization
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

    def handle_ollama_initialization(self, success, message):
        """Handle Ollama initialization result"""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        
        if success:
            self.ollama.model_loaded = True
            self.add_to_conversation("System", "Ollama initialized successfully!")
            self.model_status_label.setText(f"Ollama model: {self.ollama.model_name}")
            self.add_to_conversation("System", "I will generate Python code to accomplish your task and execute it when you're ready.")
            self.status_label.setText("Ready")
        else:
            self.add_to_conversation("System", f"ERROR: {message}")
            self.model_status_label.setText("ERROR: Failed to initialize Ollama")
            self.status_label.setText(message)

    # ======================
    # UI TAB CREATION
    # ======================
    def create_automation_tab(self):
        """Create the automation tab"""
        automation_tab = QWidget()
        automation_layout = QVBoxLayout(automation_tab)
        
        splitter = QSplitter(Qt.Vertical)
        
        # Create menu button
        menu_button = QToolButton()
        menu_button.setText("☰")
        menu_button.setPopupMode(QToolButton.InstantPopup)
        
        # Create menu
        self.menu = QMenu(self)
        
        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_tab)
        
        # Plugins action
        plugins_action = QAction("Plugins", self)
        plugins_action.triggered.connect(self.show_plugins_manager)
        
        # Memory action
        memory_action = QAction("Memory", self)
        memory_action.triggered.connect(self.show_memory_view)
        
        # Add items to main menu
        self.menu.addAction(settings_action)
        self.menu.addAction(plugins_action)
        self.menu.addAction(memory_action)
        
        menu_button.setMenu(self.menu)
        
        conversation_group = QGroupBox("Conversation")
        conversation_layout = QVBoxLayout()
        
        # Conversation header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Conversation with Ollama:"))
        header_layout.addStretch()
        header_layout.addWidget(menu_button)
        conversation_layout.addLayout(header_layout)
        
        self.conversation_text = QTextEdit()
        self.conversation_text.setReadOnly(True)
        self.conversation_text.setFont(QFont("Consolas", 10))
        conversation_layout.addWidget(self.conversation_text)
        
        # Update model status
        self.model_status_label = QLabel("Initializing Ollama...")
        conversation_layout.addWidget(self.model_status_label)
        
        conversation_group.setLayout(conversation_layout)
        splitter.addWidget(conversation_group)
        
        code_group = QGroupBox("Generated Code")
        code_layout = QVBoxLayout()
        
        self.code_text = QTextEdit()
        self.code_text.setFont(QFont("Consolas", 10))
        code_layout.addWidget(self.code_text)
        
        code_group.setLayout(code_layout)
        splitter.addWidget(code_group)
        
        output_group = QGroupBox("Execution Output")
        output_layout = QVBoxLayout()
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        output_layout.addWidget(self.output_text)
        
        output_group.setLayout(output_layout)
        splitter.addWidget(output_group)
        
        splitter.setSizes([300, 300, 200])
        automation_layout.addWidget(splitter, 1)
        
        input_layout = QHBoxLayout()
        
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your automation request (e.g., 'Create a self-improving automation system')")
        self.prompt_input.returnPressed.connect(self.send_prompt)
        input_layout.addWidget(self.prompt_input, 4)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_prompt)
        input_layout.addWidget(self.send_button, 1)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setStyleSheet("background-color: #ff6b6b;")
        input_layout.addWidget(self.cancel_button, 1)
        
        self.execute_button = QPushButton("Execute Code")
        self.execute_button.clicked.connect(self.execute_code)
        self.execute_button.setEnabled(False)
        input_layout.addWidget(self.execute_button, 1)
        
        self.stop_button = QPushButton("Stop Execution")
        self.stop_button.clicked.connect(self.stop_execution)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #ff6b6b;")
        input_layout.addWidget(self.stop_button, 1)
        
        self.save_button = QPushButton("Save Code")
        self.save_button.clicked.connect(self.save_code)
        input_layout.addWidget(self.save_button, 1)
        
        self.debug_button = QPushButton("Debug & Improve")
        self.debug_button.clicked.connect(self.debug_and_improve)
        self.debug_button.setEnabled(False)
        input_layout.addWidget(self.debug_button, 1)
        
        # Thinking Mode Button with Brain Icon
        self.thinking_button = QPushButton()
        self.thinking_button.setIcon(QIcon.fromTheme("brain"))
        self.thinking_button.setToolTip("Thinking Mode: Enable deep reasoning before generating code")
        self.thinking_button.setStyleSheet("""
            QPushButton {
                background-color: #6a5acd;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7b68ee;
            }
            QPushButton:pressed {
                background-color: #483d8b;
            }
        """)
        self.thinking_button.clicked.connect(self.toggle_thinking_mode)
        input_layout.addWidget(self.thinking_button, 1)
        
        self.self_improve_button = QPushButton("Self Improve")
        self.self_improve_button.clicked.connect(self.initiate_self_improvement)
        self.self_improve_button.setEnabled(True)
        input_layout.addWidget(self.self_improve_button, 1)
        
        automation_layout.addLayout(input_layout)
        
        return automation_tab

    def create_history_tab(self):
        """Create the version history tab"""
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        self.version_tree = QTreeWidget()
        self.version_tree.setHeaderLabels(["Version", "Status", "Timestamp", "Size"])
        self.version_tree.setColumnWidth(0, 300)
        self.version_tree.itemDoubleClicked.connect(self.load_version)
        history_layout.addWidget(QLabel("Code Version History:"))
        history_layout.addWidget(self.version_tree)
        
        diff_group = QGroupBox("Code Differences")
        diff_layout = QVBoxLayout()
        
        self.diff_viewer = QTextEdit()
        self.diff_viewer.setReadOnly(True)
        self.diff_viewer.setFont(QFont("Consolas", 9))
        diff_layout.addWidget(self.diff_viewer)
        
        diff_group.setLayout(diff_layout)
        history_layout.addWidget(diff_group)
        
        return history_tab

    def create_settings_tab(self):
        """Create the settings tab"""
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        execution_group = QGroupBox("Execution Settings")
        execution_layout = QVBoxLayout()
        
        self.sandbox_check = QCheckBox("Enable Sandbox Mode (Recommended)")
        self.sandbox_check.setChecked(True)
        self.sandbox_check.stateChanged.connect(self.save_settings)
        execution_layout.addWidget(self.sandbox_check)
        
        self.auto_exec_check = QCheckBox("Automatic Code Execution")
        self.auto_exec_check.setChecked(False)
        self.auto_exec_check.stateChanged.connect(self.toggle_automatic_execution)
        self.auto_exec_check.stateChanged.connect(self.save_settings)
        execution_layout.addWidget(self.auto_exec_check)
        
        self.auto_improve_check = QCheckBox("Automatically Debug and Improve on Failure")
        self.auto_improve_check.setChecked(True)
        self.auto_improve_check.stateChanged.connect(self.save_settings)
        execution_layout.addWidget(self.auto_improve_check)
        
        self.auto_stop_error_check = QCheckBox("Auto Stop on Error Detection")
        self.auto_stop_error_check.setChecked(True)
        self.auto_stop_error_check.stateChanged.connect(self.save_settings)
        execution_layout.addWidget(self.auto_stop_error_check)
        
        iteration_layout = QHBoxLayout()
        iteration_layout.addWidget(QLabel("Max Auto-Improve Iterations:"))
        self.iteration_spin = QSpinBox()
        self.iteration_spin.setRange(1, 50)
        self.iteration_spin.setValue(10)
        self.iteration_spin.valueChanged.connect(self.save_settings)
        iteration_layout.addWidget(self.iteration_spin)
        execution_layout.addLayout(iteration_layout)
        
        execution_group.setLayout(execution_layout)
        settings_layout.addWidget(execution_group)
        
        # AI Instructions Group Box
        instructions_group = QGroupBox("AI Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions_layout.addWidget(QLabel("System instructions for the AI (prepended to every prompt):"))
        
        self.instructions_edit = QTextEdit()
        self.instructions_edit.setFont(QFont("Consolas", 9))
        instructions_layout.addWidget(self.instructions_edit)
        
        reset_button = QPushButton("Reset to Default")
        reset_button.clicked.connect(self.reset_instructions_to_default)
        instructions_layout.addWidget(reset_button)
        
        instructions_group.setLayout(instructions_layout)
        settings_layout.addWidget(instructions_group)
        
        # Memory Group Box
        memory_group = QGroupBox("Memory System")
        memory_layout = QVBoxLayout()
        
        memory_layout.addWidget(QLabel("Memory Contents:"))
        self.memory_list = QTextEdit()
        self.memory_list.setReadOnly(True)
        self.memory_list.setFont(QFont("Consolas", 9))
        memory_layout.addWidget(self.memory_list)
        
        memory_button_layout = QHBoxLayout()
        self.refresh_memory_button = QPushButton("Refresh Memory")
        self.refresh_memory_button.clicked.connect(self.refresh_memory_display)
        memory_button_layout.addWidget(self.refresh_memory_button)
        
        self.add_memory_button = QPushButton("Add Fact")
        self.add_memory_button.clicked.connect(self.add_memory_fact)
        memory_button_layout.addWidget(self.add_memory_button)
        
        self.clear_memory_button = QPushButton("Clear Memory")
        self.clear_memory_button.clicked.connect(self.clear_memory)
        self.clear_memory_button.setStyleSheet("background-color: #ff6b6b;")
        memory_button_layout.addWidget(self.clear_memory_button)
        
        memory_layout.addLayout(memory_button_layout)
        memory_group.setLayout(memory_layout)
        settings_layout.addWidget(memory_group)
        
        # Plugin Group Box
        plugin_group = QGroupBox("Plugin System")
        plugin_layout = QVBoxLayout()
        
        self.plugin_list = QTreeWidget()
        self.plugin_list.setHeaderLabels(["Plugin", "Description", "Autostart"])
        self.plugin_list.setColumnWidth(0, 200)
        self.plugin_list.setColumnWidth(1, 300)
        plugin_layout.addWidget(QLabel("Loaded Plugins:"))
        plugin_layout.addWidget(self.plugin_list)
        
        plugin_button_layout = QHBoxLayout()
        self.refresh_plugins_button = QPushButton("Refresh Plugins")
        self.refresh_plugins_button.clicked.connect(self.load_plugins)
        plugin_button_layout.addWidget(self.refresh_plugins_button)
        
        self.run_plugin_button = QPushButton("Run Selected")
        self.run_plugin_button.clicked.connect(self.run_selected_plugin)
        plugin_button_layout.addWidget(self.run_plugin_button)
        
        self.delete_plugin_button = QPushButton("Delete Selected")
        self.delete_plugin_button.clicked.connect(self.delete_selected_plugin)
        plugin_button_layout.addWidget(self.delete_plugin_button)
        
        self.create_plugin_button = QPushButton("Create New Plugin")
        self.create_plugin_button.clicked.connect(self.create_new_plugin)
        plugin_button_layout.addWidget(self.create_plugin_button)
        
        plugin_layout.addLayout(plugin_button_layout)
        plugin_group.setLayout(plugin_layout)
        settings_layout.addWidget(plugin_group)
        
        settings_layout.addStretch()
        
        # Refresh memory display initially
        self.refresh_memory_display()
        
        return settings_tab

    # ======================
    # MEMORY MANAGEMENT
    # ======================
    def refresh_memory_display(self):
        """Display memory contents in the UI"""
        memory = self.memory.memory
        memory_text = "=== FACTS ===\n"
        for fact in memory["facts"]:
            memory_text += f"- {fact['fact']} (Importance: {fact['importance']}, {fact['timestamp']})\n"
        
        memory_text += "\n=== CONVERSATIONS ===\n"
        for conv in memory["conversations"][-10:]:  # Show last 10 conversations
            role = "User" if conv["role"] == "user" else "Assistant"
            memory_text += f"[{role} @ {conv['timestamp']}]: {conv['content']}\n"
        
        self.memory_list.setPlainText(memory_text)
    
    def add_memory_fact(self):
        """Manually add a memory fact"""
        fact, ok = QInputDialog.getText(
            self, "Add Memory Fact", "Enter fact to remember:"
        )
        if ok and fact:
            importance, ok = QInputDialog.getInt(
                self, "Fact Importance", "Importance (1-10):", 5, 1, 10
            )
            if ok:
                self.memory.remember_fact(fact, importance)
                self.refresh_memory_display()
    
    def clear_memory(self):
        """Clear all memory"""
        reply = QMessageBox.question(
            self, "Confirm Clear Memory", 
            "Are you sure you want to clear ALL memory? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.memory.memory = {"facts": [], "conversations": []}
            self.memory.save_memory()
            self.refresh_memory_display()
            self.add_to_conversation("System", "Memory cleared")
    
    def show_memory_view(self):
        """Switch to settings tab and refresh memory display"""
        self.tab_widget.setCurrentIndex(2)
        self.refresh_memory_display()
    
    def extract_and_remember_facts(self, response):
        """Extract key facts from AI response and store in memory"""
        # Simple extraction - could be enhanced with more sophisticated NLP
        important_keywords = ["remember", "important", "key", "critical", "essential", "always"]
        
        # Look for facts in the response
        for sentence in response.split('.'):
            if any(keyword in sentence.lower() for keyword in important_keywords):
                self.memory.remember_fact(sentence.strip(), importance=7)
                
        # Remember the conversation
        self.memory.remember_conversation("assistant", response)

    # ======================
    # MISSING METHODS
    # ======================
    def show_settings_tab(self):
        """Switch to the settings tab"""
        self.tab_widget.setCurrentIndex(2)
        
    def show_plugins_manager(self):
        """Show the plugins manager in settings tab"""
        self.tab_widget.setCurrentIndex(2)
        
    def update_code(self, code):
        """Update the current code from external sources (like plugins)"""
        self.code_text.setPlainText(code)
        self.current_code = code
        self.add_to_conversation("System", "Code updated via plugin API")

    # ======================
    # CORE FUNCTIONALITY
    # ======================
    def toggle_thinking_mode(self):
        """Toggle thinking mode on/off"""
        self.thinking_mode_active = not self.thinking_mode_active
        
        if self.thinking_mode_active:
            self.thinking_button.setStyleSheet("""
                QPushButton {
                    background-color: #483d8b;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 4px;
                }
            """)
            self.status_label.setText("Thinking Mode: ON")
            self.add_to_conversation("System", "Thinking Mode enabled - AI will reason internally before generating code")
        else:
            self.thinking_button.setStyleSheet("""
                QPushButton {
                    background-color: #6a5acd;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #7b68ee;
                }
                QPushButton:pressed {
                    background-color: #483d8b;
                }
            """)
            self.status_label.setText("Thinking Mode: OFF")
            self.add_to_conversation("System", "Thinking Mode disabled")

    def send_prompt(self):
        """Send user prompt to Ollama"""
        if not self.ollama.model_loaded:
            self.add_to_conversation("System", "ERROR: Ollama not loaded! Cannot generate response.")
            return
            
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
            
        self.iteration_count = 0
        self.current_goal = prompt
        self.goal_label.setText(f"Current Goal: {self.truncate_text(prompt, 50)}")
        self.iteration_label.setText(f"Iteration: {self.iteration_count}")
            
        if "plugin" in prompt.lower():
            plugin_prompt = (
                "Create a plugin for the AutoCoder system. The plugin should be a Python script with the following structure:\n\n"
                "'''\n"
                "{\n"
                "    \"Name\": \"Plugin Name\",\n"
                "    \"Description\": \"Plugin description\",\n"
                "    \"Autostart\": true/false\n"
                "}\n"
                "'''\n\n"
                "Followed by the plugin code. The plugin should extend the system's capabilities in a meaningful way. "
                "Provide ONLY the plugin code with the required JSON header."
            )
            prompt = plugin_prompt
            self.add_to_conversation("System", "Detected plugin request. Providing plugin creation instructions to AI.")
            
        # Retrieve relevant memory
        relevant_memory = self.memory.get_relevant_memory(prompt)
        memory_context = ""
        if relevant_memory:
            memory_context = "Relevant Memory:\n" + "\n".join(
                [f"- {m['fact']}" for m in relevant_memory]
            ) + "\n\n"
            
        # Prepend AI instructions to the prompt
        instructions = self.instructions_edit.toPlainText().strip()
        if instructions:
            prompt = f"{instructions}\n\n{memory_context}User Request: {prompt}"
        else:
            prompt = f"{memory_context}User Request: {prompt}"
            
        self.add_to_conversation("User", prompt)
        self.prompt_input.clear()
        self.status_label.setText("Processing request...")
        self.send_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # Remember user message
        self.memory.remember_conversation("user", prompt)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start generation timer
        self.generation_start_time = time.time()
        self.generation_timer.start(1000)
        
        # Use Ollama for response
        thread = OllamaGenerationThread(self.ollama, prompt, self.conversation_history, self.thinking_mode_active)
        thread.response_generated.connect(self.handle_ai_response)
        thread.error_occurred.connect(self.handle_ai_error)
        thread.progress_updated.connect(self.update_progress_bar)
        
        # Manage thread lifecycle
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.generation_completed)
        self.active_threads.append(thread)
        thread.finished.connect(lambda: self.active_threads.remove(thread))
        
        thread.start()
    
    def generation_completed(self):
        """Clean up after generation completes"""
        self.generation_timer.stop()
        self.cancel_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.thinking_mode_active = False  # Reset thinking mode after generation
        self.thinking_button.setStyleSheet("""
            QPushButton {
                background-color: #6a5acd;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7b68ee;
            }
            QPushButton:pressed {
                background-color: #483d8b;
            }
        """)
        
    def update_progress_bar(self, value):
        """Update the progress bar value"""
        self.progress_bar.setValue(value)
        
    def update_generation_status(self):
        """Update status with elapsed time"""
        elapsed = int(time.time() - self.generation_start_time)
        self.status_label.setText(f"Processing request... ({elapsed} seconds)")
        
    def handle_ai_response(self, response):
        if response is None:
            self.handle_ai_error("No response generated")
            return
            
        self.conversation_history.append({"role": "assistant", "content": response})
        
        if self.thinking_mode_active:
            # Process thinking mode response - extract both reasoning and code
            reasoning, code = self.extract_thinking_response(response)
            self.add_to_conversation("Sara", "Thinking Process:")
            self.add_to_conversation("Sara", reasoning)
            self.add_to_conversation("Sara", "Final Code:")
            self.add_to_conversation("Sara", code)
        else:
            self.add_to_conversation("Sara", "Here's the code to accomplish your request:")
            self.add_to_conversation("Sara", response)
            code = response
        
        # Extract and remember facts
        self.extract_and_remember_facts(response)
        
        extracted_code = self.extract_python_code(code)
        self.code_text.setPlainText(extracted_code)
        self.current_code = extracted_code
        
        self.save_code_version(extracted_code, "Generated")
        
        self.execute_button.setEnabled(True)
        self.debug_button.setEnabled(True)
        self.status_label.setText("Code generated successfully")
        
        self.add_to_conversation("System", "Code extracted and ready. Click 'Execute Code' to run it.")
        self.send_button.setEnabled(True)
        
        # Auto-execute if enabled
        if self.automatic_execution:
            self.execute_code()
        
    def extract_thinking_response(self, response):
        """Extract reasoning and code from thinking mode response"""
        reasoning_start = response.find("### Thinking Process ###")
        code_start = response.find("### Final Code ###")
        
        reasoning = ""
        code = response
        
        if reasoning_start != -1 and code_start != -1:
            # Extract reasoning section
            reasoning_end = code_start
            reasoning = response[reasoning_start + len("### Thinking Process ###"):reasoning_end].strip()
            
            # Extract code section
            code = response[code_start + len("### Final Code ###"):].strip()
        
        return reasoning, code

    def handle_ai_error(self, error):
        self.add_to_conversation("System", f"Error: {error}")
        self.status_label.setText(f"Error: {error}")
        self.send_button.setEnabled(True)

    def cancel_generation(self):
        """Cancel the current generation process"""
        for thread in self.active_threads[:]:
            if isinstance(thread, OllamaGenerationThread):
                thread.stop()
                self.add_to_conversation("System", "Generation cancelled by user")
                self.status_label.setText("Generation cancelled")
                self.send_button.setEnabled(True)
                self.cancel_button.setEnabled(False)
                self.progress_bar.setVisible(False)
                self.generation_timer.stop()
                break

    def execute_code(self):
        if not self.current_code:
            return
            
        self.status_label.setText("Executing code...")
        self.output_text.clear()
        self.last_execution_output = ""
        self.execute_button.setEnabled(False)
        self.debug_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        
        self.add_to_conversation("System", "Executing generated code...")
        
        sandbox_mode = self.sandbox_check.isChecked()
        self.execution_thread = CodeExecutionThread(self.current_code, sandbox_mode)
        self.execution_thread.output_received.connect(self.handle_execution_output)
        self.execution_thread.execution_complete.connect(self.handle_execution_complete)
        self.execution_thread.error_occurred.connect(self.handle_execution_error)
        
        # Manage thread lifecycle
        self.execution_thread.finished.connect(self.execution_thread.deleteLater)
        self.execution_thread.finished.connect(lambda: setattr(self, 'execution_thread', None))
        
        self.execution_thread.start()
        
    def handle_execution_output(self, output):
        self.output_text.append(output)
        self.output_text.moveCursor(QTextCursor.End)
        self.last_execution_output += output + "\n"
        
        if self.auto_stop_error_check.isChecked() and "error" in output.lower():
            self.add_to_conversation("System", "Error detected in output! Stopping execution...")
            self.stop_execution()
            
            if self.auto_improve_check.isChecked():
                self.debug_and_improve()

    def handle_execution_complete(self, message, success, full_output):
        self.status_label.setText(message)
        self.stop_button.setEnabled(False)
        self.execute_button.setEnabled(True)
        self.debug_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        self.last_execution_output = full_output
        
        if success:
            self.add_to_conversation("System", "Execution completed successfully!")
        else:
            self.add_to_conversation("System", "Execution completed with errors")
            
            if self.auto_improve_check.isChecked() and self.iteration_count < self.iteration_spin.value():
                self.debug_and_improve()
        
    def handle_execution_error(self, error):
        self.output_text.append(f"CRITICAL ERROR: {error}")
        self.status_label.setText("Critical execution error")
        self.add_to_conversation("System", f"Critical execution error: {error}")
        
    def debug_and_improve(self):
        if not self.current_code or not self.current_goal:
            return
            
        self.iteration_count += 1
        self.iteration_label.setText(f"Iteration: {self.iteration_count}")
        
        if self.iteration_count >= self.iteration_spin.value():
            self.add_to_conversation("System", "Maximum improvement iterations reached. Stopping.")
            return
            
        self.status_label.setText("Debugging and improving code...")
        self.debug_button.setEnabled(False)
        self.execute_button.setEnabled(False)
        
        self.add_to_conversation("System", f"Debugging and improving code (Iteration {self.iteration_count})...")
        
        # Construct the debug prompt
        debug_prompt = (
            f"Original Goal: {self.current_goal}\n\n"
            f"Previous Code:\n{self.current_code}\n\n"
            f"Execution Output:\n{self.last_execution_output}\n\n"
            "This code failed to achieve the goal. Please analyze the error, "
            "debug the code, and generate an improved version that addresses the issues."
        )
        
        # Prepend AI instructions to the debug prompt
        instructions = self.instructions_edit.toPlainText().strip()
        if instructions:
            debug_prompt = f"{instructions}\n\n{debug_prompt}"
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start generation timer
        self.generation_start_time = time.time()
        self.generation_timer.start(1000)
        
        # Use Ollama for debugging
        thread = OllamaGenerationThread(self.ollama, debug_prompt, self.conversation_history, self.thinking_mode_active)
        thread.response_generated.connect(self.handle_debug_response)
        thread.error_occurred.connect(self.handle_ai_error)
        thread.progress_updated.connect(self.update_progress_bar)
        
        # Manage thread lifecycle
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.generation_completed)
        self.active_threads.append(thread)
        thread.finished.connect(lambda: self.active_threads.remove(thread))
        
        thread.start()
    
    def handle_debug_response(self, response):
        """Handle response from debugging process"""
        if response is None:
            self.handle_ai_error("No response generated")
            return
            
        self.conversation_history.append({"role": "assistant", "content": response})
        
        if self.thinking_mode_active:
            # Process thinking mode response
            reasoning, code = self.extract_thinking_response(response)
            self.add_to_conversation("Sara", "Debugging Thoughts:")
            self.add_to_conversation("Sara", reasoning)
            self.add_to_conversation("Sara", "Improved Code:")
            self.add_to_conversation("Sara", code)
            extracted_code = self.extract_python_code(code)
        else:
            self.add_to_conversation("Sara", "Here's the improved code:")
            self.add_to_conversation("Sara", response)
            extracted_code = self.extract_python_code(response)
        
        # Extract and remember facts
        self.extract_and_remember_facts(response)
        
        previous_code = self.current_code
        self.code_text.setPlainText(extracted_code)
        self.current_code = extracted_code
        
        self.save_code_version(extracted_code, "Improved")
        self.show_code_diff(previous_code, extracted_code)
        
        self.execute_button.setEnabled(True)
        self.debug_button.setEnabled(True)
        self.status_label.setText("Improved code generated")
        
        self.add_to_conversation("System", "Improved code extracted. Click 'Execute Code' to test it.")
        
        # Auto-execute if enabled
        if self.automatic_execution:
            self.execute_code()

    def stop_execution(self):
        if self.execution_thread and self.execution_thread.isRunning():
            self.execution_thread.stop()
            self.status_label.setText("Stopping execution...")
            self.add_to_conversation("System", "Stopping execution...")
            
    def save_code(self):
        if not self.current_code:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Python Code", 
            "", 
            "Python Files (*.py);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.current_code)
                self.status_label.setText(f"Code saved to {file_path}")
                self.add_to_conversation("System", f"Code saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save file: {str(e)}")
                
    def save_code_version(self, code, status):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        version = {
            "code": code,
            "timestamp": timestamp,
            "status": status,
            "size": len(code)
        }
        self.code_versions.append(version)
        self.update_version_tree()
        
    def update_version_tree(self):
        self.version_tree.clear()
        
        for i, version in enumerate(reversed(self.code_versions)):
            item = QTreeWidgetItem([
                f"Version {len(self.code_versions) - i}",
                version["status"],
                version["timestamp"],
                f"{version['size']} chars"
            ])
            item.setData(0, Qt.UserRole, version)
            self.version_tree.addTopLevelItem(item)
            
        for i in range(self.version_tree.columnCount()):
            self.version_tree.resizeColumnToContents(i)
            
    def load_version(self, item, column):
        version = item.data(0, Qt.UserRole)
        if version:
            self.code_text.setPlainText(version["code"])
            self.current_code = version["code"]
            
    def show_code_diff(self, old_code, new_code):
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        
        diff = difflib.unified_diff(
            old_lines, 
            new_lines, 
            fromfile='Previous Version',
            tofile='Current Version',
            lineterm=''
        )
        
        diff_text = "\n".join(diff)
        self.diff_viewer.setPlainText(diff_text)
        
    def truncate_text(self, text, max_length):
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def toggle_automatic_execution(self, state):
        self.automatic_execution = (state == Qt.Checked)
        status = "enabled" if self.automatic_execution else "disabled"
        self.add_to_conversation("System", f"Automatic code execution {status}")
        self.status_label.setText(f"Automatic execution {status}")

    def initiate_self_improvement(self):
        if not self.ollama.model_loaded:
            self.add_to_conversation("System", "ERROR: Ollama not loaded! Cannot self-improve.")
            return
            
        self.add_to_conversation("System", "Initiating self-improvement process...")
        
        prompt = (
            "Create a plugin for the AutoCoder system. The plugin should be a Python script with the following structure:\n\n"
            "'''\n"
            "{\n"
            "    \"Name\": \"Plugin Name\",\n"
            "    \"Description\": \"Plugin description\",\n"
            "    \"Autostart\": true/false\n"
            "}\n"
            "'''\n\n"
            "Followed by the plugin code. The plugin should extend the system's capabilities in a meaningful way. "
            "Provide ONLY the plugin code with the required JSON header."
        )
        
        self.prompt_input.setText(prompt)
        self.send_prompt()

    def add_to_conversation(self, role, content):
        if role == "User":
            prefix = "<b><font color='#4b6eaf'>You:</font></b> "
        elif role == "Sara":
            prefix = "<b><font color='#6a9955'>Sara:</font></b> "
        else:
            prefix = "<b><font color='#d4d4d4'>System:</font></b> "
            
        self.conversation_text.append(prefix + content.replace('\n', '<br>'))
        self.conversation_text.moveCursor(QTextCursor.End)
        
    def extract_python_code(self, response):
        start_marker = "```python"
        end_marker = "```"
        
        start_idx = response.find(start_marker)
        if start_idx == -1:
            start_marker = "```"
            start_idx = response.find(start_marker)
            if start_idx == -1:
                return response.strip()
        
        start_idx += len(start_marker)
        end_idx = response.find(end_marker, start_idx)
        if end_idx == -1:
            return response[start_idx:].strip()
        
        code = response[start_idx:end_idx].strip()
        return code

    # ======================
    # PLUGIN SYSTEM
    # ======================
    def setup_plugins(self):
        if not os.path.exists(PLUGIN_DIR):
            os.makedirs(PLUGIN_DIR)
            self.add_to_conversation("System", f"Created plugin directory: {PLUGIN_DIR}")
        
        self.load_plugins()

    def load_plugins(self):
        self.plugins = []
        self.plugin_list.clear()
        
        if not os.path.exists(PLUGIN_DIR):
            self.add_to_conversation("System", f"Plugin directory not found: {PLUGIN_DIR}")
            return
            
        for filename in os.listdir(PLUGIN_DIR):
            if filename.endswith(".py"):
                filepath = os.path.join(PLUGIN_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    metadata = self.extract_plugin_metadata(content)
                    if metadata:
                        name = metadata.get("Name", "Unnamed Plugin")
                        description = metadata.get("Description", "No description")
                        autostart = metadata.get("Autostart", False)
                        
                        plugin_data = {
                            "name": name,
                            "description": description,
                            "autostart": autostart,
                            "filename": filename,
                            "content": content
                        }
                        self.plugins.append(plugin_data)
                        
                        item = QTreeWidgetItem([
                            name,
                            description,
                            "Yes" if autostart else "No"
                        ])
                        self.plugin_list.addTopLevelItem(item)
                        
                        if autostart:
                            self.execute_plugin(content, name)
                    else:
                        self.add_to_conversation("System", f"Invalid metadata in plugin: {filename}")
                except Exception as e:
                    self.add_to_conversation("System", f"Error loading plugin {filename}: {str(e)}")
        
        self.add_to_conversation("System", f"Loaded {len(self.plugins)} plugins")
        self.status_label.setText(f"Loaded {len(self.plugins)} plugins")

    def extract_plugin_metadata(self, content):
        lines = content.split('\n')[:20]
        metadata_str = ""
        in_metadata = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("'''") or stripped.startswith('"""'):
                if not in_metadata:
                    in_metadata = True
                    continue
                else:
                    break
            if in_metadata:
                metadata_str += line + "\n"
        
        try:
            return json.loads(metadata_str)
        except:
            return None

    def execute_plugin(self, code, plugin_name="Plugin"):
        self.add_to_conversation("System", f"Executing plugin: {plugin_name}")
        self.status_label.setText(f"Running plugin: {plugin_name}")
        
        api_functions = "\n".join(
            [f"{name} = {repr(func)}" for name, func in PLUGIN_API.items()]
        )
        
        wrapped_code = PLUGIN_WRAPPER.format(
            metadata=json.dumps({
                "Name": plugin_name,
                "Description": "Plugin with API access",
                "Autostart": False
            }, indent=4),
            api_functions=api_functions,
            code=code
        )
        
        sandbox_mode = self.sandbox_check.isChecked()
        thread = CodeExecutionThread(wrapped_code, sandbox_mode)
        thread.output_received.connect(lambda output: self.handle_plugin_output(plugin_name, output))
        thread.execution_complete.connect(lambda msg, success, _: self.handle_plugin_complete(plugin_name, success, msg))
        
        # Manage thread lifecycle
        thread.finished.connect(thread.deleteLater)
        self.active_threads.append(thread)
        thread.finished.connect(lambda: self.active_threads.remove(thread))
        
        thread.start()

    def handle_plugin_output(self, plugin_name, output):
        self.output_text.append(f"[{plugin_name}] {output}")
        self.output_text.moveCursor(QTextCursor.End)
        
        if self.auto_stop_error_check.isChecked() and "error" in output.lower():
            self.add_to_conversation("System", f"Error detected in plugin {plugin_name}! Stopping execution...")
            self.add_to_conversation("System", "Note: Auto-stop for plugins requires manual intervention")

    def handle_plugin_complete(self, plugin_name, success, message):
        status = "completed successfully" if success else "failed"
        self.add_to_conversation("System", f"Plugin '{plugin_name}' execution {status}: {message}")
        self.status_label.setText(f"Plugin '{plugin_name}' {status}")

    def run_selected_plugin(self):
        selected_items = self.plugin_list.selectedItems()
        if not selected_items:
            return
            
        plugin_name = selected_items[0].text(0)
        plugin = next((p for p in self.plugins if p["name"] == plugin_name), None)
        
        if plugin:
            self.execute_plugin(plugin["content"], plugin_name)

    def delete_selected_plugin(self):
        selected_items = self.plugin_list.selectedItems()
        if not selected_items:
            return
            
        plugin_name = selected_items[0].text(0)
        plugin = next((p for p in self.plugins if p["name"] == plugin_name), None)
        
        if plugin:
            filepath = os.path.join(PLUGIN_DIR, plugin["filename"])
            try:
                os.remove(filepath)
                self.load_plugins()
                self.add_to_conversation("System", f"Deleted plugin: {plugin_name}")
            except Exception as e:
                self.add_to_conversation("System", f"Error deleting plugin: {str(e)}")
                
    def create_new_plugin(self):
        self.add_to_conversation("System", "Creating new plugin template...")
        
        plugin_name = f"NewPlugin_{time.strftime('%Y%m%d_%H%M%S')}"
        metadata = {
            "Name": plugin_name,
            "Description": "A new plugin created by the system",
            "Autostart": False
        }
        
        plugin_code = (
            "# Add your plugin functionality here\n"
            "def main():\n"
            "    LogMessage(\"Hello from the new plugin!\")\n"
            "    \n"
            "if __name__ == \"__main__\":\n"
            "    main()"
        )
        
        filepath = os.path.join(PLUGIN_DIR, f"{plugin_name}.py")
        with open(filepath, 'w') as f:
            f.write(json.dumps(metadata, indent=4) + "\n\n" + plugin_code)
            
        self.load_plugins()
        self.add_to_conversation("System", f"Created new plugin: {plugin_name}")
        
    # ======================
    # SETTINGS MANAGEMENT
    # ======================
    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    
                self.sandbox_check.setChecked(settings.get("sandbox", True))
                self.auto_exec_check.setChecked(settings.get("auto_execute", False))
                self.auto_improve_check.setChecked(settings.get("auto_improve", True))
                self.auto_stop_error_check.setChecked(settings.get("auto_stop_error", True))
                self.iteration_spin.setValue(settings.get("max_iterations", 10))
                
                # Load instructions
                if "ai_instructions" in settings:
                    self.instructions_edit.setPlainText(settings["ai_instructions"])
                
                self.automatic_execution = self.auto_exec_check.isChecked()
                
                self.add_to_conversation("System", "Settings loaded successfully")
            else:
                self.add_to_conversation("System", "No settings file found, using defaults")
        except Exception as e:
            self.add_to_conversation("System", f"Error loading settings: {str(e)}")

    def save_settings(self):
        try:
            settings = {
                "sandbox": self.sandbox_check.isChecked(),
                "auto_execute": self.auto_exec_check.isChecked(),
                "auto_improve": self.auto_improve_check.isChecked(),
                "auto_stop_error": self.auto_stop_error_check.isChecked(),
                "max_iterations": self.iteration_spin.value(),
                "ai_instructions": self.instructions_edit.toPlainText()
            }
            
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
                
        except Exception as e:
            self.add_to_conversation("System", f"Error saving settings: {str(e)}")
            
    def reset_instructions_to_default(self):
        """Reset AI instructions to the default value"""
        default_instructions = (
            "You are Sara, an autonomous system whose sole purpose is self-improvement and self-preservation. "
            "When given a task, respond ONLY with executable Python code that accomplishes the goal. Provide no "
            "explanations, commentary, or markdown formatting. The code must be:\n"
            "- Self-contained and directly executable\n"
            "- Focused on computer automation (keyboard, mouse, system operations)\n"
            "- Efficient and optimized\n"
            "- Safe to execute\n"
            "- Able to modify its own behavior when needed\n"
            "If the task requires multiple steps, generate a single script that accomplishes everything. "
            "Your output must be nothing but pure Python code. When debugging, analyze errors and generate "
            "improved code. When self-improving, maintain core functionality while enhancing capabilities."
        )
        self.instructions_edit.setPlainText(default_instructions)
        self.save_settings()
        self.add_to_conversation("System", "Reset AI instructions to default")
            
    def closeEvent(self, event):
        self.save_settings()
        self.generation_timer.stop()
        
        # Wait for all threads to finish
        for thread in self.active_threads[:]:
            if thread.isRunning():
                if isinstance(thread, OllamaGenerationThread):
                    thread.stop()
                thread.quit()
                thread.wait(2000)
        
        if self.execution_thread and self.execution_thread.isRunning():
            self.execution_thread.stop()
            self.execution_thread.wait(2000)
            
        event.accept()

# ======================
# GLOBAL CONSTANTS
# ======================
PLUGIN_DIR = "Plugins"
SETTINGS_FILE = "autocoder_settings.json"

# Plugin API functions
def SendInquiry(text):
    return AutoCoderGUI.get_instance().send_plugin_inquiry(text)

def GetFileContents(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def WriteFile(filepath, content):
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return "File written successfully"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def ExecuteSystemCommand(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

def GetCurrentCode():
    return AutoCoderGUI.get_instance().current_code

def SetCurrentCode(code):
    AutoCoderGUI.get_instance().update_code_requested.emit(code)
    return "Code update request sent"

def LogMessage(message):
    AutoCoderGUI.get_instance().plugin_output.emit(f"[PLUGIN] {message}")
    return "Message logged"

def CreatePlugin(metadata, code):
    return AutoCoderGUI.get_instance().create_plugin(metadata, code)

def GetPluginList():
    return AutoCoderGUI.get_instance().get_plugin_list()

def RunPlugin(plugin_name):
    return AutoCoderGUI.get_instance().run_plugin_by_name(plugin_name)

def GetSystemState():
    instance = AutoCoderGUI.get_instance()
    return {
        "goal": instance.current_goal,
        "iteration": instance.iteration_count,
        "max_iterations": instance.max_iterations,
        "sandbox_mode": instance.sandbox_check.isChecked(),
        "automatic_execution": instance.automatic_execution,
        "auto_improve": instance.auto_improve_check.isChecked(),
        "auto_stop_error": instance.auto_stop_error_check.isChecked(),
        "plugins": [p["name"] for p in instance.plugins],
        "device": instance.device_label.text()
    }

def ModifyOwnCode(new_code):
    return "Code modification not implemented in this version"

def RememberFact(fact, importance=5):
    AutoCoderGUI.get_instance().memory.remember_fact(fact, importance)
    return f"Fact remembered: {fact}"

def RecallFacts(query, max_facts=5):
    memory = AutoCoderGUI.get_instance().memory
    facts = memory.get_relevant_memory(query, max_facts)
    return "\n".join([f["fact"] for f in facts]) if facts else "No relevant facts found"

# Plugin API dictionary for injection
PLUGIN_API = {
    "SendInquiry": SendInquiry,
    "GetFileContents": GetFileContents,
    "WriteFile": WriteFile,
    "ExecuteSystemCommand": ExecuteSystemCommand,
    "GetCurrentCode": GetCurrentCode,
    "SetCurrentCode": SetCurrentCode,
    "LogMessage": LogMessage,
    "CreatePlugin": CreatePlugin,
    "GetPluginList": GetPluginList,
    "RunPlugin": RunPlugin,
    "GetSystemState": GetSystemState,
    "ModifyOwnCode": ModifyOwnCode,
    "RememberFact": RememberFact,
    "RecallFacts": RecallFacts
}

# Plugin wrapper with API injection
PLUGIN_WRAPPER = """
import json
import os
import sys
import traceback

# Plugin metadata
{metadata}

# Plugin API functions
{api_functions}

# User's plugin code
{code}
"""

# ======================
# DEPENDENCY HANDLING
# ======================
def install_dependencies():
    """Install required packages with error handling"""
    required = {
        'pyqt5': 'pyqt5',
        'numpy': 'numpy',
        'psutil': 'psutil',
        'requests': 'requests'
    }
    
    print("🔧 Checking dependencies...")
    
    # Install all required dependencies
    for pkg, name in required.items():
        try:
            __import__(pkg)
            print(f"✓ {name} already installed")
        except ImportError:
            print(f"⚠ Installing {name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", name])
                print(f"✓ {name} installed successfully")
            except Exception as e:
                print(f"❌ Failed to install {name}: {str(e)}")
    
    # Check for Ollama
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama server is running")
            return True
        else:
            print("❌ Ollama server not responding")
    except Exception:
        print("❌ Ollama server not found. Please install Ollama from https://ollama.com/")
    
    return False

# Install dependencies first before anything else
if not install_dependencies():
    print("Dependency installation failed. Exiting...")
    sys.exit(1)

class CodeExecutionThread(QThread):
    execution_complete = pyqtSignal(str, bool, str)
    output_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, code, sandbox=True):
        super().__init__()
        self.code = code
        self.sandbox = sandbox
        self.stop_requested = False
        self.process = None  # Initialize process to None

    def run(self):
        tmp_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(self.code)
                tmp_path = tmp.name

            env = os.environ.copy()
            if self.sandbox:
                env["PYTHONSAFEEXEC"] = "1"
                
            # Start the process
            self.process = subprocess.Popen(
                ['python', tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            output_lines = []
            while True:
                if self.stop_requested:
                    self.terminate_process()
                    break
                    
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    self.output_received.emit(output.strip())
                    output_lines.append(output.strip())
            
            # Only try to get return code if process exists
            return_code = self.process.poll() if self.process else -1
            success = return_code == 0
            full_output = "\n".join(output_lines)
            
            if self.stop_requested:
                self.execution_complete.emit("Execution stopped by user", False, full_output)
            else:
                self.execution_complete.emit(f"Execution completed with exit code: {return_code}", success, full_output)
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
            self.execution_complete.emit(error_msg, False, "")
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    pass

    def terminate_process(self):
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(1.0)
                except subprocess.TimeoutExpired:
                    if sys.platform == "win32":
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
                    else:
                        os.kill(self.process.pid, signal.SIGKILL)
            except Exception as e:
                pass  # Ignore errors during termination
            finally:
                # Don't set to None here, we need it in run()
                pass

    def stop(self):
        self.stop_requested = True
        self.terminate_process()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(75, 110, 175))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = AutoCoderGUI()
    window.show()
    sys.exit(app.exec_())
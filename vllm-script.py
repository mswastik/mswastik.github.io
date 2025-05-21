"""
vllm-nexus-gui-hybrid.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import socket
import json
import threading
from vllm import AsyncLLMEngine, SamplingParams
import subprocess
import GPUtil
import time
import os
import torch
import psutil
import mmap
import sys
from datetime import datetime
import re
import pynvml

class VLLMServerGUI:
    def __init__(self, master):
        self.master = master
        master.title("VLLM-DRAM-VRAM Server Manager")
        
        # Configuration parameters storage
        self.config = {
            'model_path': '',
            'ip': self.get_local_ip(),
            'port': 8000,
            'gpu_count': 1,
            'mem_ratio': 95,  # Increase VRAM utilization
            'max_tokens': 4096,  # Increase max tokens
            'kv_dtype': 'float16',
            'block_size': 16,
            'max_blocks': '',
            'calculate_scales': True,
            'max_model_len': 4096,  # Reduce max_model_len to save memory
            # Memory offload related configuration
            'enable_memory_offload': True,  # Enable memory offload by default
            'memory_offload_ratio': 70,  # Increase memory offload ratio
            'memory_channels': 4,
            'reserved_memory': 20
        }
        
        # Server process
        self.server_process = None
        
        # API address
        self.api_address = None
        
        # Main interface layout
        self.create_widgets()
        
        # Load configuration
        self.load_config()
        
        # Professional monitoring flag
        self.monitoring = True
        # Start GPU monitoring thread
        threading.Thread(target=self.update_gpu_stats, daemon=True).start()
        
    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP
    
    def create_widgets(self):
        # Basic Configuration Area
        self.config_frame = ttk.LabelFrame(self.master, text="Basic Configuration")
        self.config_frame.pack(padx=10, pady=5, fill='x')
        
        # Model Path
        ttk.Label(self.config_frame, text="Model Path:").grid(row=0, column=0)
        self.model_path_entry = ttk.Entry(self.config_frame, width=50)
        self.model_path_entry.grid(row=0, column=1)
        self.model_path_entry.insert(0, self.config['model_path'])
        ttk.Button(self.config_frame, text="Browse", command=self.select_model_path).grid(row=0, column=2)
        
        # Add Save Config button and Recommend Settings button
        save_config_button = ttk.Button(self.config_frame, text="Save Config", command=self.save_config_with_message)
        save_config_button.grid(row=0, column=3, padx=5)
        recommend_button = ttk.Button(self.config_frame, text="Recommend Settings", command=self.recommend_settings)
        recommend_button.grid(row=0, column=4, padx=5)
        
        # IP Address
        ttk.Label(self.config_frame, text="IP Address:").grid(row=1, column=0)
        self.ip_entry = ttk.Entry(self.config_frame)
        self.ip_entry.grid(row=1, column=1, sticky='w')
        self.ip_entry.insert(0, self.config['ip'])
        
        # Port
        ttk.Label(self.config_frame, text="Port:").grid(row=2, column=0)
        self.port_entry = ttk.Entry(self.config_frame)
        self.port_entry.grid(row=2, column=1, sticky='w')
        self.port_entry.insert(0, str(self.config['port']))
        
        # GPU Count
        ttk.Label(self.config_frame, text="GPU Count:").grid(row=3, column=0)
        self.gpu_count_var = tk.StringVar(value=str(self.config['gpu_count']))
        gpu_count_combo = ttk.Combobox(self.config_frame, textvariable=self.gpu_count_var,
                                     values=["1", "2", "3", "4"], width=5)
        gpu_count_combo.grid(row=3, column=1, sticky='w')
        
        # VRAM Ratio
        ttk.Label(self.config_frame, text="VRAM Ratio (%):").grid(row=4, column=0)
        self.mem_ratio_entry = ttk.Entry(self.config_frame)
        self.mem_ratio_entry.grid(row=4, column=1, sticky='w')
        self.mem_ratio_entry.insert(0, str(self.config['mem_ratio']))
        
        # Max Tokens
        ttk.Label(self.config_frame, text="Max Tokens:").grid(row=5, column=0)
        self.max_tokens_var = tk.StringVar(value=str(self.config['max_tokens']))
        ttk.Entry(self.config_frame, textvariable=self.max_tokens_var, width=8).grid(row=5, column=1)
        ttk.Label(self.config_frame, text="(Response tokens should not be less than total sequence length)", foreground="gray").grid(row=6, column=0, columnspan=2, sticky='w')
        
        # Max Sequence Length
        ttk.Label(self.config_frame, text="Max Sequence Length:").grid(row=5, column=2)
        self.max_model_len_var = tk.StringVar(value=str(self.config['max_model_len']))
        max_model_len_combo = ttk.Combobox(self.config_frame, textvariable=self.max_model_len_var,
                                         values=["2048", "4096", "8192", "16384", "32768", "65536"], width=8)
        max_model_len_combo.grid(row=5, column=3)
        ttk.Label(self.config_frame, text="(Please choose appropriate parameters based on hardware conditions)", foreground="gray").grid(row=6, column=2, columnspan=2, sticky='w')
        
        # KV Cache Configuration
        cache_frame = ttk.LabelFrame(self.config_frame, text="KV Cache Configuration")
        cache_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Cache Precision
        ttk.Label(cache_frame, text="Cache Precision:").grid(row=0, column=0)
        self.kv_dtype_var = tk.StringVar(value=self.config['kv_dtype'])
        ttk.Combobox(cache_frame, textvariable=self.kv_dtype_var,
                    values=["float16", "float32"], width=8).grid(row=0, column=1)
        
        # Cache Block Size
        ttk.Label(cache_frame, text="Block Size (tokens):").grid(row=0, column=2)
        self.block_size_var = tk.StringVar(value=str(self.config['block_size']))
        ttk.Entry(cache_frame, textvariable=self.block_size_var, width=8).grid(row=0, column=3)
        
        # Max Cache Blocks
        ttk.Label(cache_frame, text="Max Blocks:").grid(row=1, column=0)
        self.max_blocks_var = tk.StringVar(value=str(self.config['max_blocks']))
        ttk.Entry(cache_frame, textvariable=self.max_blocks_var, width=8).grid(row=1, column=1)
        ttk.Label(cache_frame, text="(Leave blank for auto)").grid(row=1, column=2)
        
        # Dynamic Scaling Option
        self.calculate_scales_var = tk.BooleanVar(value=self.config['calculate_scales'])
        ttk.Checkbutton(cache_frame, text="Enable Dynamic Scaling", 
                       variable=self.calculate_scales_var).grid(row=1, column=3)
        
        # Add Advanced Performance Settings Area
        self.create_advanced_settings()
        
        # Monitoring Panel
        monitor_frame = ttk.LabelFrame(self.master, text="GPU Monitoring")
        monitor_frame.pack(padx=10, pady=5, fill='both', expand=True)
        
        # GPU Status Display
        columns = ('GPU', 'VRAM Usage', 'GPU Usage', 'Temperature', 'Power', 'KV Cache Hit Rate')
        self.gpu_tree = ttk.Treeview(monitor_frame, columns=columns, show='headings')
        for col in columns:
            self.gpu_tree.heading(col, text=col)
            self.gpu_tree.column(col, width=100)
        self.gpu_tree.pack(fill='both', expand=True)
        
        # Status Display Area
        self.status_text = tk.Text(monitor_frame, height=10)
        self.status_text.pack(fill='both')
        
        # Server Control Buttons
        button_frame = ttk.Frame(self.config_frame)
        button_frame.grid(row=8, column=0, columnspan=3, pady=5)
        ttk.Button(button_frame, text="Start Server", command=self.start_server).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Stop Server", command=self.stop_server).grid(row=0, column=1, padx=5)
        
        # API Address Display
        self.api_label = ttk.Label(self.config_frame, text="API Address:")
        self.api_label.grid(row=9, column=0, columnspan=3)
        
        # Add Memory Offload Configuration Frame
        offload_frame = ttk.LabelFrame(self.config_frame, text="Memory Offload Configuration")
        offload_frame.grid(row=10, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Enable Memory Offload Option
        self.enable_offload_var = tk.BooleanVar(value=self.config['enable_memory_offload'])
        ttk.Checkbutton(offload_frame, text="Enable Memory Offload", 
                       variable=self.enable_offload_var).grid(row=0, column=0)
        
        # Memory Channel Count
        ttk.Label(offload_frame, text="Memory Channels:").grid(row=0, column=1)
        self.memory_channels_var = tk.StringVar(value=str(self.config['memory_channels']))
        ttk.Combobox(offload_frame, textvariable=self.memory_channels_var,
                    values=["2", "4", "8", "16"], width=5).grid(row=0, column=2)
        
        # Memory Offload Ratio
        ttk.Label(offload_frame, text="Memory Offload Ratio (%):").grid(row=1, column=0)
        self.memory_offload_ratio_var = tk.StringVar(value=str(self.config['memory_offload_ratio']))
        ttk.Entry(offload_frame, textvariable=self.memory_offload_ratio_var, width=5).grid(row=1, column=1)
        
        # Reserved System Memory Ratio
        ttk.Label(offload_frame, text="Reserved System Memory (%):").grid(row=1, column=2)
        self.reserved_memory_var = tk.StringVar(value=str(self.config['reserved_memory']))
        ttk.Entry(offload_frame, textvariable=self.reserved_memory_var, width=5).grid(row=1, column=3)
        
        # Add Advanced Description
        ttk.Label(offload_frame, text="(Enabling this allows loading large models exceeding VRAM, but reduces inference speed)", 
                 foreground="gray").grid(row=2, column=0, columnspan=4, sticky='w')
        
        # Add "Check Compatibility" button
        self.check_compatibility_button = ttk.Button(
            self.config_frame, 
            text="Check Compatibility", 
            command=self.check_model_compatibility
        )
        self.check_compatibility_button.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # Add Performance Monitoring Panel
        self.add_performance_monitoring()
    
    def select_model_path(self):
        path = filedialog.askdirectory()
        if path:
            self.config['model_path'] = path
            self.model_path_entry.delete(0, tk.END)  # Clear current content
            self.model_path_entry.insert(0, path)    # Insert new path
            
    def start_server(self):
        """Start VLLM Server"""
        if not self.config['model_path']:
            messagebox.showerror("Error", "Please select a model path first")
            return
        
        if hasattr(self, 'server_process') and self.server_process and self.server_process.poll() is None:
            messagebox.showinfo("Hint", "Server is already running")
            return
        
        # Check model compatibility
        if not self.check_model_compatibility():
            if not messagebox.askokcancel("Warning", "Model compatibility check found potential issues, continue starting server?"):
                return
        
        # Clean GPU memory
        self.clean_gpu_memory()
        
        # Set environment variables to avoid memory fragmentation issues
        env = os.environ.copy()

        # Apply CUDA memory chunk size from advanced settings
        cuda_split_size = self.config.get('advanced_cuda_split_size', 128)  # Default 128MB
        env['PYTORCH_CUDA_ALLOC_CONF'] = f'expandable_segments:True,max_split_size_mb:{cuda_split_size}'
        self.status_text.insert(tk.END, f"CUDA memory chunk size: {cuda_split_size}MB\n")

        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(self.config['gpu_count'])])
        env['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
        env['MKL_NUM_THREADS'] = '4'  # Limit MKL threads

        # Add performance optimization environment variables
        env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'  # Optimize CUDA connections
        env['NCCL_P2P_DISABLE'] = '1'  # For single GPU, disabling P2P may improve performance
        env['CUDA_AUTO_BOOST'] = '1'  # Enable GPU auto boost frequency
        env['VLLM_USE_ASYNC_CUDA_MALLOC'] = '1'  # Use asynchronous CUDA memory allocation
        # Get system memory size
        system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        # Choose whether to enable memory-efficient linear layers based on hardware
        if system_memory > 16:  # Only enable if system memory is sufficient
            env['VLLM_ENABLE_MEMORY_EFFICIENT_LINEAR'] = '1'  # Enable memory-efficient linear layers
        
        # Log startup information
        self.status_text.insert(tk.END, "\n===== Starting Server =====\n")
        self.status_text.insert(tk.END, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.status_text.insert(tk.END, f"Model Path: {self.config['model_path']}\n")
        self.status_text.insert(tk.END, f"GPU Count: {self.config['gpu_count']}\n")
        self.status_text.insert(tk.END, f"VRAM Ratio: {self.config['mem_ratio']}%\n")
        
        # Check GPU monitoring thread
        if not self.monitoring:
            self.monitoring = True
            threading.Thread(target=self.update_gpu_stats, daemon=True).start()
        
        # Save configuration
        self.save_config()
        
        # Pre-allocate memory space to prevent out-of-memory during runtime
        self.preallocate_memory_buffer()
        
        # Initialize KV cache monitoring
        self.kv_cache_hits = 0
        self.kv_cache_misses = 0
        
        # Check if memory offload is needed
        if self.config['enable_memory_offload']:
            try:
                self.status_text.insert(tk.END, "Setting up memory offload...\n")
                
                # Calculate model size
                model_size = self.estimate_model_size()
                
                # Get available VRAM
                available_vram = self.get_available_vram(use_ratio=self.config['mem_ratio'] / 100)
                
                self.status_text.insert(tk.END, f"Model size: {model_size:.2f}GB, Available VRAM: {available_vram:.2f}GB\n")
                
                # Calculate memory size to offload
                offload_ratio = self.config['memory_offload_ratio'] / 100
                initial_offload_size = model_size * offload_ratio
                
                self.status_text.insert(tk.END, f"Offloading {initial_offload_size:.2f}GB to system memory (Ratio: {self.config['memory_offload_ratio']}%)\n")
                
                # Set up memory-mapped file
                self.setup_memory_offload(model_size, offload_ratio)
                
                # Check VLLM supported parameters
                self.status_text.insert(tk.END, "Checking VLLM supported parameters...\n")
                
                # Calculate available system memory (considering reserved ratio)
                available_memory = self.get_available_system_memory()
                reserved_ratio = self.config['reserved_memory'] / 100
                safe_memory = available_memory * (1 - reserved_ratio)
                    
                # Get actual allocated memory size
                actual_offload_size = 0
                if hasattr(self, 'mm') and self.mm:
                    try:
                        # Get memory-mapped file size
                        map_file = os.path.join(os.getcwd(), "model_offload", "model_offload.bin")
                        if os.path.exists(map_file):
                            actual_offload_size = os.path.getsize(map_file) / (1024 * 1024 * 1024)
                            self.status_text.insert(tk.END, f"Actual allocated memory-mapped size: {actual_offload_size:.2f}GB\n")
                    except Exception as e:
                        self.status_text.insert(tk.END, f"Failed to get memory-mapped size: {str(e)}\n")
                
                # Dynamically adjust required memory size
                min_required_size = min(18, model_size * 0.8)  # At least 80% of model size
                
                if actual_offload_size < min_required_size:
                    self.status_text.insert(tk.END, f"Warning: Actual allocated memory-mapped size is insufficient {min_required_size:.1f}GB, model may not load\n")
                    if not messagebox.askokcancel("Warning", 
                        f"Actual allocated memory-mapped size is only {actual_offload_size:.2f}GB, recommended at least {min_required_size:.1f}GB.\nContinue?"):
                        return False
                
                # Calculate reasonable swap space size - dynamically adjust based on model size
                # For small models (<10GB), use smaller swap space
                if model_size < 10:
                    swap_size = max(2.0, model_size * 0.1)
                else:
                    # For large models, use larger swap space
                    swap_size = max(4.0, model_size * 0.15)
                
                # Ensure it doesn't exceed 20% of safe memory
                swap_size = min(swap_size, safe_memory * 0.2)
                
                # Calculate reasonable CPU offload size - dynamically adjust based on model size and available VRAM
                available_vram = self.get_available_vram(use_ratio=self.config['mem_ratio'] / 100)
                
                # If model size exceeds available VRAM, calculate the portion to offload
                if model_size > available_vram:
                    # Size to offload = Model size - Available VRAM + extra buffer (1GB)
                    min_offload_size = model_size - available_vram + 1.0
                    # Ensure at least 60% of the model is offloaded
                    offload_size = max(min_offload_size, model_size * 0.6)
                else:
                    # If the model can fit entirely in VRAM, still offload a portion for stability
                    offload_size = model_size * 0.3
                
                # Ensure it doesn't exceed 70% of safe memory
                offload_size = min(offload_size, safe_memory * 0.7)
                
                # Calculate total memory usage
                total_mem_usage = swap_size + offload_size
                mem_usage_ratio = total_mem_usage / safe_memory * 100
                    
                self.status_text.insert(tk.END, f"Available system memory: {available_memory:.2f}GB, Safe memory: {safe_memory:.2f}GB\n")
                self.status_text.insert(tk.END, f"Calculated swap space: {swap_size:.2f}GB, CPU Offload: {offload_size:.2f}GB\n")
                self.status_text.insert(tk.END, f"Total memory usage: {total_mem_usage:.2f}GB ({mem_usage_ratio:.1f}% of safe memory)\n")
                
                # Ensure max_num_batched_tokens is greater than or equal to max_num_seqs
                max_tokens = max(self.config['max_tokens'], 256)  # Ensure at least 256
                
                # Build command
                cmd = [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', str(self.config['mem_ratio'] / 100),
                    '--max-num-batched-tokens', str(max_tokens),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half'  # Force half precision
                ]
                
                # Add max blocks (if specified)
                if self.config['max_blocks']:
                    cmd.extend(['--num-gpu-blocks', self.config['max_blocks']])
                
                # Add swap space parameter
                swap_param = f"{swap_size:.2f}"  # Remove GiB unit, use only number
                cmd.extend(['--swap-space', swap_param])
                self.status_text.insert(tk.END, f"Adding swap space parameter: --swap-space {swap_param} (GB)\n")
                
                # Add CPU offload parameter
                offload_param = f"{offload_size:.2f}"  # Remove GB unit, use only number
                cmd.extend(['--cpu-offload-gb', offload_param])
                self.status_text.insert(tk.END, f"Adding CPU offload parameter: --cpu-offload-gb {offload_param} (GB)\n")
                
                # Add enforce-eager mode to avoid out-of-memory during CUDA graph capture phase
                cmd.append('--enforce-eager')
                self.status_text.insert(tk.END, "Adding enforce-eager parameter: --enforce-eager\n")
                    
                self.status_text.insert(tk.END, f"Memory offload enabled, available CPU memory: {safe_memory:.2f}GB\n")
                    
                # Log full command
                cmd_str = ' '.join(cmd)
                self.status_text.insert(tk.END, f"Full command: {cmd_str}\n")
                self.status_text.see(tk.END)
                
            except Exception as e:
                self.status_text.insert(tk.END, f"Error setting up memory offload: {str(e)}\n")
                import traceback
                self.status_text.insert(tk.END, traceback.format_exc())
                if not messagebox.askokcancel("Error", 
                    f"Error setting up memory offload: {str(e)}\nContinue starting server (without memory offload)?"):
                    return
                
                # If memory offload setup fails, use basic command
                max_tokens = max(self.config['max_tokens'], 256)  # Ensure at least 256
                cmd = [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', str(self.config['mem_ratio'] / 100),
                    '--max-num-batched-tokens', str(max_tokens),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',  # Force half precision
                    '--enforce-eager'  # Add enforce-eager mode to avoid out-of-memory during CUDA graph capture phase
                ]
        else:
            # If memory offload is not needed, use basic command
            max_tokens = max(self.config['max_tokens'], 256)  # Ensure at least 256
            cmd = [
                'vllm', 'serve',
                self.config['model_path'],
                '--host', self.config['ip'],
                '--port', str(self.config['port']),
                '--tensor-parallel-size', str(self.config['gpu_count']),
                '--gpu-memory-utilization', str(self.config['mem_ratio'] / 100),
                '--max-num-batched-tokens', str(max_tokens),
                '--block-size', str(self.config['block_size']),
                '--max-model-len', str(self.config['max_model_len']),
                '--dtype', 'half',  # Force half precision
                '--enforce-eager'  # Add enforce-eager mode to avoid out-of-memory during CUDA graph capture phase
            ]
        
        # Add performance optimization parameters
        performance_args = [
            '--max-num-seqs', '32',  # Increase max sequences
            '--disable-log-stats',  # Disable statistics logging, reduce overhead
            '--kv-cache-dtype', 'auto',  # Use auto-select KV cache precision
            '--trust-remote-code'  # Trust remote code, support more models
        ]
        
        # Apply batch size from advanced settings
        batch_size = self.config.get('advanced_batch_size', 16)  # Default 16
        performance_args.extend(['--max-num-batched-tokens', str(max(batch_size * 256, max_tokens))])
        self.status_text.insert(tk.END, f"Batch size: {batch_size}\n")

        # Add memory bandwidth optimization parameters
        if int(self.block_size_var.get()) < 32:
            # If block size is less than 32, suggest increasing to 32 to improve memory bandwidth utilization
            self.status_text.insert(tk.END, f"Note: Current block size ({self.block_size_var.get()}) is small, may affect memory bandwidth utilization\n")
            self.status_text.insert(tk.END, "Suggest using a larger block size (32-64) to improve memory bandwidth utilization\n")

        # Check for Flash Attention support
        if self.check_flash_attention_support():
            performance_args.append('--enable-chunked-prefill')
            self.status_text.insert(tk.END, "Enabling chunked prefill optimization\n")
        
        # Add performance parameters to command
        cmd.extend(performance_args)
        
        # Start server asynchronously
        try:
            self.status_text.insert(tk.END, "Starting server process...\n")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env  # Use modified environment variables
            )
            
            # Wait for a short period, check if the process exits immediately
            time.sleep(1)
            if self.server_process.poll() is not None:
                # Process exited, get output
                output, _ = self.server_process.communicate()
                error_msg = f"Failed to start server: {output.decode()}"
                self.status_text.insert(tk.END, f"{error_msg}\n")
                
                # Attempt fallback method
                return self.fallback_start_server(error_msg)
            
            # Start monitoring thread
            threading.Thread(target=self.monitor_server_output).start()

            # Update API address
            # Note: GET /v1 returning 404 is normal, please use specific API endpoint supporting POST for requests
            api_base = f"http://{self.config['ip']}:{self.config['port']}/v1"
            self.api_label.config(text=f"API Address: {api_base}")
            self.status_text.insert(tk.END, f"\nServer starting...\nAPI Address: {api_base}\n")
            self.status_text.see(tk.END)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to start server: {str(e)}"
            self.status_text.insert(tk.END, f"{error_msg}\n")
            import traceback
            self.status_text.insert(tk.END, traceback.format_exc())
            
            # Attempt fallback method
            return self.fallback_start_server(error_msg)
    
    def stop_server(self):
        try:
            # First stop all monitoring threads
            self.monitoring = False
            # Wait for a short period for threads to exit
            time.sleep(0.5)
            
            if hasattr(self, 'server_process') and self.server_process and self.server_process.poll() is None:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                    self.status_text.insert(tk.END, "\nServer stopped.\n")
                except subprocess.TimeoutExpired:
                    self.status_text.insert(tk.END, "\nStopping server timed out, but server may have stopped.\n")
            else:
                self.status_text.insert(tk.END, "\nServer is not running.\n")
                
            # Clean up memory offload resources
            self.cleanup_memory_offload()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop server: {str(e)}")
        finally:
            # Ensure monitoring flag is set to False
            self.monitoring = False
            # Disable auto-tuning
            if hasattr(self, 'auto_tune_var'):
                self.auto_tune_var.set(False)
            self.api_label.config(text="API Address: Server not started")
    
    def cleanup_memory_offload(self):
        """Clean up memory offload resources"""
        try:
            # Clean up memory buffer
            self.cleanup_memory_buffer()
            
            # Clean up multi-channel loader
            if hasattr(self, 'multi_channel_loader') and self.multi_channel_loader is not None:
                try:
                    # Call loader's close method
                    if hasattr(self.multi_channel_loader, 'close'):
                        self.multi_channel_loader.close()
                    self.multi_channel_loader = None
                    self.status_text.insert(tk.END, "Multi-channel loader closed\n")
                except Exception as e:
                    self.status_text.insert(tk.END, f"Error closing multi-channel loader: {str(e)}\n")
            elif hasattr(self, 'channel_loaders'):
                # Compatible with older code versions
                for loader in self.channel_loaders:
                    if hasattr(loader, 'mm') and loader.mm:
                        loader.mm.close()
                    if hasattr(loader, 'mm_file') and loader.mm_file:
                        loader.mm_file.close()
                self.channel_loaders = []
            
            # Clean up memory map
            if hasattr(self, 'mm') and self.mm:
                self.mm.close()
                self.mm = None
                
            if hasattr(self, 'mm_file') and self.mm_file:
                self.mm_file.close()
                self.mm_file = None
            
            self.status_text.insert(tk.END, "Memory-mapped resources released\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Error releasing memory-mapped resources: {str(e)}\n")
    
    def monitor_server_output(self):
        """Monitor server output and detect errors"""
        error_patterns = [
            # Out of memory errors
            (r"CUDA out of memory", "GPU Out of Memory"),
            (r"OutOfMemoryError", "Out of Memory"),
            (r"OOM", "Out of Memory"),
            # Model loading errors
            (r"Error loading model", "Model Loading Error"),
            (r"Failed to load", "Model Loading Failed"),
            # Parameter errors
            (r"ValueError", "Parameter Error"),
            (r"TypeError", "Type Error"),
            # Permission errors
            (r"PermissionError", "Permission Error"),
            # Network errors
            (r"ConnectionError", "Connection Error"),
            (r"Address already in use", "Port Already in Use"),
            # General errors
            (r"Error:", "Error Occurred"),
            (r"Exception:", "Exception Occurred"),
            (r"Traceback", "Program Crash")
        ]
        
        # Token generation pattern
        token_pattern = r"Processed (\d+) tokens"
        
        # Record startup time
        start_time = time.time()
        error_detected = False
        error_message = ""
        server_started = False
        
        while True:
            if not hasattr(self, 'server_process') or self.server_process is None:
                self.status_text.insert(tk.END, "Server process does not exist\n")
                break
                
            if self.server_process.poll() is not None:
                self.status_text.insert(tk.END, f"Server process exited with code: {self.server_process.poll()}\n")
                break
                
            try:
                output = self.server_process.stdout.readline()
                if not output:
                    time.sleep(0.1)
                    continue
                    
                output_text = output.decode(errors='replace')
                self.status_text.insert(tk.END, output_text)
                self.status_text.see(tk.END)
                
                # Check for token generation information
                token_match = re.search(token_pattern, output_text)
                if token_match:
                    tokens = int(token_match.group(1))
                    self.update_token_count(tokens)
                
                # Check for error messages
                for pattern, error_type in error_patterns:
                    if re.search(pattern, output_text, re.IGNORECASE):
                        error_detected = True
                        error_message = f"{error_type}: {output_text.strip()}"
                        self.status_text.insert(tk.END, f"Error detected: {error_type}\n")
                        break
                        
                # If error detected, wait for a while to collect more logs, then try to recover
                if error_detected:
                    # Continue reading some output to get more error information
                    for _ in range(10):  # Read up to 10 lines of extra output
                        try:
                            more_output = self.server_process.stdout.readline()
                            if more_output:
                                more_text = more_output.decode(errors='replace')
                                self.status_text.insert(tk.END, more_text)
                                error_message += "\n" + more_text.strip()
                        except:
                            break
                        time.sleep(0.1)
                    
                    # If it's an out of memory error, try fallback startup method
                    if "Out of Memory" in error_message:
                        self.status_text.insert(tk.END, "Out of memory error detected, attempting fallback startup method...\n")
                        # Stop current process
                        try:
                            self.server_process.terminate()
                            self.server_process.wait(timeout=5)
                        except:
                            pass
                        # Attempt to start using fallback method
                        self.fallback_start_server(error_message)
                        return
                    # If port is already in use, try a different port
                    elif "Port Already in Use" in error_message:
                        self.status_text.insert(tk.END, "Port already in use detected, attempting to use a different port...\n")
                        # Stop current process
                        try:
                            self.server_process.terminate()
                            self.server_process.wait(timeout=5)
                        except:
                            pass
                        # Attempt to use a different port
                        self.config['port'] += 1
                        self.status_text.insert(tk.END, f"Attempting to use new port: {self.config['port']}\n")
                        self.start_server()
                        return
                    else:
                        # Other errors, display error message and ask user if they want to try fallback method
                        if messagebox.askokcancel("Error", f"An error occurred while starting the server:\n{error_message}\n\nAttempt to start using fallback method?"):
                            # Stop current process
                            try:
                                self.server_process.terminate()
                                self.server_process.wait(timeout=5)
                            except:
                                pass
                            # Attempt to start using fallback method
                            self.fallback_start_server(error_message)
                        return
                
                # Check if successfully started (usually API related information appears within a few seconds)
                if "API server" in output_text and time.time() - start_time > 5:
                    self.status_text.insert(tk.END, "Server started successfully\n")
                    server_started = True
                    
                    # After server starts successfully, perform automatic performance optimization
                    if not hasattr(self, 'performance_optimized') or not self.performance_optimized:
                        threading.Thread(target=self.auto_optimize_performance, daemon=True).start()
                        self.performance_optimized = True
                
            except Exception as e:
                self.status_text.insert(tk.END, f"Error monitoring server output: {str(e)}\n")
                time.sleep(1)
    
    def update_gpu_stats(self):
        while self.monitoring:
            try:
                gpus = GPUtil.getGPUs()
                self.gpu_tree.delete(*self.gpu_tree.get_children())
                for gpu in gpus:
                    # Use nvidia-smi to get power information
                    try:
                        power_info = subprocess.run(
                            ['nvidia-smi', f'--id={gpu.id}', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                            capture_output=True,
                            text=True
                        )
                        power_draw = power_info.stdout.strip()
                    except:
                        power_draw = "N/A"
                    
                    self.gpu_tree.insert('', 'end', values=(
                        gpu.id,
                        f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB",
                        f"{gpu.load*100:.1f}%",
                        f"{gpu.temperature}°C",
                        f"{power_draw}W" if power_draw and power_draw != "N/A" else "N/A",
                        "0.0%"  # KV Cache Hit Rate not supported yet
                    ))
                time.sleep(2)
            except Exception as e:
                self.status_text.insert(tk.END, f"GPU monitoring error: {e}\n")
                self.status_text.see(tk.END)
                time.sleep(5)
    
    def get_gpu_stats(self):
        """Get GPU statistics, return list of dictionaries"""
        try:
            # Use pynvml library instead of executing nvidia-smi command
            pynvml.nvmlInit()
            
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_stats = []
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = f"{utilization.gpu} %"
                mem_util = f"{utilization.memory} %"
                
                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Get power consumption
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_draw = f"{power:.1f} W"
                except:
                    power_draw = "N/A"
                
                gpu_stat = {
                    'utilization.gpu': gpu_util,
                    'utilization.memory': mem_util,
                    'temperature.gpu': f"{temp} C",
                    'power.draw': power_draw
                }
                gpu_stats.append(gpu_stat)
            
            pynvml.nvmlShutdown()
            return gpu_stats
        except ImportError:
            # If pynvml is not installed, return a simulated status message and log a warning
            self.status_text.insert(tk.END, "Warning: pynvml not installed, cannot get GPU information. Please install with pip install nvidia-ml-py3.\n")
            # Return a dictionary with default values to prevent program crash
            return [{'utilization.gpu': '0 %', 'utilization.memory': '0 %', 'temperature.gpu': '0 C', 'power.draw': 'N/A'}]
        except Exception as e:
            # Log error but return an empty result set instead of raising an exception
            self.status_text.insert(tk.END, f"Error getting GPU statistics: {str(e)}\n")
            return []
    
    def load_config(self):
        try:
            with open('server_config.json', 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
                
                # Update values on the interface
                self.model_path_entry.delete(0, tk.END)
                self.model_path_entry.insert(0, self.config['model_path'])
                
                self.ip_entry.delete(0, tk.END)
                self.ip_entry.insert(0, self.config['ip'])
                
                self.port_entry.delete(0, tk.END)
                self.port_entry.insert(0, str(self.config['port']))
                
                self.gpu_count_var.set(str(self.config['gpu_count']))
                
                self.mem_ratio_entry.delete(0, tk.END)
                self.mem_ratio_entry.insert(0, str(self.config['mem_ratio']))
                
                self.max_tokens_var.set(str(self.config['max_tokens']))
                
                self.max_model_len_var.set(str(self.config['max_model_len']))  # Load max_model_len
                
                # Load memory offload configuration
                if 'enable_memory_offload' in self.config:
                    self.enable_offload_var.set(self.config['enable_memory_offload'])
                if 'memory_channels' in self.config:
                    self.memory_channels_var.set(str(self.config['memory_channels']))
                if 'memory_offload_ratio' in self.config:
                    self.memory_offload_ratio_var.set(str(self.config['memory_offload_ratio']))
                if 'reserved_memory' in self.config:
                    self.reserved_memory_var.set(str(self.config['reserved_memory']))
                
        except FileNotFoundError:
            pass

    def save_config(self):
        with open('server_config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def save_config_with_message(self):
        # First call update_config to ensure configuration is updated
        if self.update_config():
            # Save configuration
            self.save_config()
            messagebox.showinfo("Success", "Configuration saved to server_config.json")

    def select_calibrated_model(self):
        path = filedialog.askdirectory(title="Select Calibrated Model Directory")
        if path:
            self.calibrated_model_var.set(path)
            self.config['calibrated_model'] = path
            
    def check_fp8_support(self):
        try:
            if not torch.cuda.is_available():
                return False
            capability = torch.cuda.get_device_capability()
            # Requires Ampere or newer architecture (compute capability >= 8.0)
            return capability[0] >= 8
        except Exception as e:
            print(f"Failed to check FP8 support: {e}")
            return False
            
    def run_calibration(self):
        if not self.check_fp8_support():
            messagebox.showerror("Error", "Current GPU does not support FP8 quantization")
            return
            
        if not self.config['model_path']:
            messagebox.showerror("Error", "Please select a model path first")
            return
            
        # Generate calibration script
        calibration_script = f"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.transformers import oneshot

# Load model
model = AutoModelForCausalLM.from_pretrained("{self.config['model_path']}", 
                                            device_map="auto", 
                                            torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("{self.config['model_path']}")

# Configure calibration parameters
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

# Quantization configuration
recipe = '''
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
'''

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save quantized model
SAVE_DIR = "{os.path.basename(self.config['model_path'])}-FP8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
"""
        
        # Save and run calibration script
        with open("run_calibration.py", "w") as f:
            f.write(calibration_script)
            
        # Detect OS, use appropriate method to start process
        try:
            if sys.platform == 'win32':
                # Windows system
                subprocess.Popen(["python", "run_calibration.py"],
                                cwd=os.getcwd(),
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # Linux/Mac system
                subprocess.Popen(["python", "run_calibration.py"],
                                cwd=os.getcwd())
                        
            messagebox.showinfo("Calibration", "Calibration process started, please wait for completion...")
        except Exception as e:
            self.status_text.insert(tk.END, f"Failed to start calibration process: {str(e)}\n")
            messagebox.showerror("Error", f"Failed to start calibration process: {str(e)}")
    
    def get_available_system_memory(self):
        """Get available system memory (GB)"""
        mem = psutil.virtual_memory()
        # Return available memory (GB)
        return mem.available / (1024 * 1024 * 1024)
    
    def get_available_vram(self, use_ratio=None):
        """Get available VRAM (GB)"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0
            
            # If using multiple GPUs, calculate total VRAM
            if self.config['gpu_count'] > 1:
                total_vram = sum([gpu.memoryTotal for gpu in gpus[:self.config['gpu_count']]])
            else:
                total_vram = gpus[0].memoryTotal
                
            # Convert to GB and apply VRAM ratio
            ratio = use_ratio if use_ratio is not None else (self.config['mem_ratio'] / 100)
            return total_vram * ratio / 1024
        except Exception as e:
            self.status_text.insert(tk.END, f"Error getting VRAM information: {e}\n")
            return 0
    
    def estimate_model_size(self):
        """Estimate model size (GB)"""
        try:
            # Simple estimation: check total size of .bin files in model directory
            model_path = self.config['model_path']
            total_size = 0
            
            # Check for model.safetensors file
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                total_size = os.path.getsize(safetensors_path)
                self.status_text.insert(tk.END, f"Found model.safetensors file, size: {total_size/(1024*1024*1024):.2f}GB\n")
                # Convert to GB
                return total_size / (1024 * 1024 * 1024)
            
            # Check for pytorch_model.bin file
            pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                total_size = os.path.getsize(pytorch_model_path)
                self.status_text.insert(tk.END, f"Found pytorch_model.bin file, size: {total_size/(1024*1024*1024):.2f}GB\n")
                # Convert to GB
                return total_size / (1024 * 1024 * 1024)
            
            # If it's a sharded model, calculate the size of all shards
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.bin') or file.endswith('.safetensors'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        self.status_text.insert(tk.END, f"Found model file: {file}, size: {file_size/(1024*1024*1024):.2f}GB\n")
            
            # If no model files found, use default value
            if total_size == 0:
                self.status_text.insert(tk.END, "No model files found, using default value 29.5GB\n")
                return 29.5  # Default value is 29.5GB
            
            # Convert to GB
            model_size_gb = total_size / (1024 * 1024 * 1024)
            self.status_text.insert(tk.END, f"Estimated total model size: {model_size_gb:.2f}GB\n")
            return model_size_gb
        except Exception as e:
            self.status_text.insert(tk.END, f"Error estimating model size: {e}\n")
            # Return default value
            return 29.5  # Default value is 29.5GB

    def setup_memory_offload(self, model_size, offload_ratio):
        """Set up memory offload function"""
        if not self.config['enable_memory_offload']:
            return False
            
        try:
            # Calculate the portion to offload to memory
            offload_size = model_size * offload_ratio
            
            self.status_text.insert(tk.END, f"Offloading {offload_size:.2f}GB to system memory (Ratio: {offload_ratio*100:.0f}%)\n")
            
            # Create memory-mapped file directory
            offload_dir = os.path.join(os.getcwd(), "model_offload")
            os.makedirs(offload_dir, exist_ok=True)
            
            # Create memory-mapped file
            map_file = os.path.join(offload_dir, "model_offload.bin")
            
            # Convert to bytes
            offload_size_bytes = int(offload_size * 1024 * 1024 * 1024)
            
            # Check for sufficient disk space
            disk_usage = psutil.disk_usage(os.getcwd())
            if disk_usage.free < offload_size_bytes:
                self.status_text.insert(tk.END, f"Warning: Insufficient disk space, {offload_size:.2f}GB required, but only {disk_usage.free/(1024*1024*1024):.2f}GB available\n")
                return False
                
            # Get system memory information
            mem = psutil.virtual_memory()
            available_memory = mem.available / (1024 * 1024 * 1024)  # Available memory (GB)
            
            # Ensure at least 2GB of system memory is reserved
            safe_memory = available_memory - 2.0
            
            # Check for sufficient memory
            if safe_memory < offload_size:
                # Adjust size to 90% of available safe memory
                adjusted_size = safe_memory * 0.9
                self.status_text.insert(tk.END, f"Warning: Insufficient available memory, {offload_size:.2f}GB required, but only {safe_memory:.2f}GB safe available memory\n")
                self.status_text.insert(tk.END, f"Automatically adjusting offload size to {adjusted_size:.2f}GB (90% of safe memory)\n")
                offload_size = adjusted_size
                offload_size_bytes = int(offload_size * 1024 * 1024 * 1024)
            
            # Create memory-mapped file
            self.status_text.insert(tk.END, f"Creating memory-mapped file, size: {offload_size:.2f}GB...\n")
            
            # Log memory usage
            mem_before = psutil.virtual_memory()
            self.status_text.insert(tk.END, f"System memory before creation: Used {mem_before.percent}% ({mem_before.used/1024/1024/1024:.2f}GB/{mem_before.total/1024/1024/1024:.2f}GB)\n")
            
            # Use fallocate to pre-allocate file space (if available)
            try:
                import subprocess
                self.status_text.insert(tk.END, f"Attempting to use fallocate to quickly allocate {offload_size:.2f}GB space...\n")
                result = subprocess.run(['fallocate', '-l', f"{offload_size_bytes}", map_file], 
                                      check=True, capture_output=True)
                self.status_text.insert(tk.END, "Successfully pre-allocated space using fallocate\n")
                
                # Verify file size
                actual_size = os.path.getsize(map_file)
                self.status_text.insert(tk.END, f"Verifying file size: {actual_size/(1024*1024*1024):.2f}GB\n")
                
                if actual_size < offload_size_bytes * 0.99:  # Allow 1% error
                    self.status_text.insert(tk.END, f"Warning: Insufficient file size, will use traditional method for allocation\n")
                    os.remove(map_file)  # Delete incomplete file
                    raise Exception("Insufficient file size")
                    
            except Exception as e:
                self.status_text.insert(tk.END, f"fallocate failed: {str(e)}, will use traditional method for allocation\n")
                
                # Traditional method: block writing
                with open(map_file, "wb") as f:
                    # Write zeros to allocate space
                    chunk_size = 1024 * 1024 * 128  # Reduce to 128MB chunks to lower memory pressure
                    remaining = offload_size_bytes
                    
                    try:
                        while remaining > 0:
                            # Check memory status every 512MB written, more frequent checks
                            if (offload_size_bytes - remaining) % (512*1024*1024) < chunk_size:
                                mem_check = psutil.virtual_memory()
                                # If available memory is below 1.5GB, stop writing
                                if mem_check.available < 1.5 * 1024 * 1024 * 1024:
                                    self.status_text.insert(tk.END, f"Warning: Available memory below 1.5GB, stopping further memory allocation\n")
                                    break
                            
                            write_size = min(chunk_size, remaining)
                            f.write(b'\0' * write_size)
                            remaining -= write_size
                            # Update progress
                            progress = (offload_size_bytes - remaining) / offload_size_bytes * 100
                            self.status_text.delete("end-2l", "end-1l")  # Delete previous progress line
                            self.status_text.insert(tk.END, f"Creating memory-mapped file: {progress:.1f}% ({(offload_size_bytes-remaining)/(1024*1024*1024):.2f}GB/{offload_size:.2f}GB)\n")
                            self.status_text.see(tk.END)
                            
                            # Add small delay to allow system to release memory
                            time.sleep(0.01)
                            
                    except MemoryError:
                        self.status_text.insert(tk.END, f"Out of memory, cannot complete mapping file creation\n")
                        # Record allocated size
                        actual_size = offload_size_bytes - remaining
                        self.status_text.insert(tk.END, f"Allocated {actual_size/(1024*1024*1024):.2f}GB\n")
                        # Truncate file to written size
                        f.flush()
                        f.truncate(actual_size)
            
            # Log memory usage
            mem_after = psutil.virtual_memory()
            self.status_text.insert(tk.END, f"System memory after creation: Used {mem_after.percent}% ({mem_after.used/1024/1024/1024:.2f}GB/{mem_after.total/1024/1024/1024:.2f}GB)\n")
            
            # Verify final file size
            final_size = os.path.getsize(map_file)
            self.status_text.insert(tk.END, f"Final memory-mapped file size: {final_size/(1024*1024*1024):.2f}GB\n")
            
            # No longer strictly require 18GB, dynamically adjust based on model size
            min_required_size = min(18, model_size * 0.8)  # At least 80% of model size
            
            if final_size < min_required_size * 1024 * 1024 * 1024:
                self.status_text.insert(tk.END, f"Warning: Memory-mapped file size is insufficient {min_required_size:.1f}GB, model may not load\n")
                if not messagebox.askokcancel("Warning", 
                    f"Memory-mapped file size is only {final_size/(1024*1024*1024):.2f}GB, recommended at least {min_required_size:.1f}GB.\nContinue?"):
                    return False
                
            self.status_text.insert(tk.END, "Memory-mapped file creation complete\n")
            
            # Create memory map
            self.mm_file = open(map_file, "r+b")
            self.mm = mmap.mmap(self.mm_file.fileno(), 0)
            
            # Use user-set memory channel count, no longer auto-increase
            channels = self.config['memory_channels']
            self.status_text.insert(tk.END, f"Using user-set memory channel count: {channels}\n")
            
            self.setup_multi_channel_loader()
            
            # Create configuration file
            offload_config = {
                'enabled': True,
                'offload_dir': offload_dir,
                'offload_ratio': offload_ratio,
                'channels': channels,
                'reserved_memory': self.config['reserved_memory'] / 100,
                'actual_size_gb': final_size/(1024*1024*1024)
            }
            
            offload_config_path = os.path.join(offload_dir, "offload_config.json")
            with open(offload_config_path, 'w') as f:
                json.dump(offload_config, f, indent=4)
            
            self.status_text.insert(tk.END, f"Memory offload configuration saved to {offload_config_path}\n")
            
            return True
        except Exception as e:
            self.status_text.insert(tk.END, f"Error setting up memory offload: {str(e)}\n")
            import traceback
            self.status_text.insert(tk.END, traceback.format_exc())
            return False
    
    def setup_multi_channel_loader(self):
        """Set up multi-channel loader"""
        class MultiChannelLoader:
            def __init__(self, memory_map, num_channels=4, cache_size=32):  # Add cache_size parameter
                self.memory_map = memory_map
                self.num_channels = num_channels
                self.channel_locks = [threading.Lock() for _ in range(num_channels)]
                self.channel_positions = [0] * num_channels
                self.channel_usage = [0] * num_channels  # Record usage count for each channel
                self.channel_last_access = [time.time()] * num_channels  # Record last access time for each channel
                self.cache = {}  # Simple memory cache
                self.cache_hits = 0
                self.cache_misses = 0
                self.max_cache_size = cache_size  # Use passed cache size
                self.prefetch_queue = []  # Prefetch queue
                self.prefetch_lock = threading.Lock()
                self.prefetch_thread_running = True
                # Start prefetch thread
                threading.Thread(target=self._prefetch_worker, daemon=True).start()
                
            def read_chunk(self, offset, size, channel_id=None):
                # Check cache
                cache_key = (offset, size)
                if cache_key in self.cache:
                    self.cache_hits += 1
                    # Update cache access time
                    self.cache[cache_key]['last_access'] = time.time()
                    return self.cache[cache_key]['data']
                
                self.cache_misses += 1
                
                # If no channel specified, choose the best channel
                if channel_id is None:
                    channel_id = self._get_best_channel(offset)
                    
                with self.channel_locks[channel_id]:
                    # Record access time
                    self.channel_last_access[channel_id] = time.time()
                    
                    # If current position is close to requested offset, reduce seek time
                    if abs(self.channel_positions[channel_id] - offset) < 1024*1024:  # If within 1MB range
                        # Already close to target position, read directly
                        pass
                    else:
                        # Need to reposition
                        self.memory_map.seek(offset)
                    
                    data = self.memory_map.read(size)
                    self.channel_positions[channel_id] = offset + size
                    self.channel_usage[channel_id] += 1
                    
                    # Update cache
                    if len(self.cache) >= self.max_cache_size:
                        # Delete oldest cache item
                        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['last_access'])
                        del self.cache[oldest_key]
                    
                    self.cache[cache_key] = {
                        'data': data,
                        'last_access': time.time()
                    }
                    
                    # Predictive prefetching - prefetch next possible block
                    next_offset = offset + size
                    self.prefetch(next_offset, size)
                    
                    return data
                    
            def _get_best_channel(self, target_offset):
                # Prioritize channels with close positions, then consider usage frequency
                best_channel = 0
                best_score = float('inf')
                
                for i in range(self.num_channels):
                    # Calculate position proximity score
                    position_score = abs(self.channel_positions[i] - target_offset) / (1024*1024)  # In MB
                    
                    # Calculate usage frequency score
                    usage_score = self.channel_usage[i] * 0.1
                    
                    # Calculate time score (less recently used is better)
                    time_score = -10 * (time.time() - self.channel_last_access[i])
                    
                    # Combined score (lower is better)
                    total_score = position_score + usage_score + time_score
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_channel = i
                
                return best_channel
                    
            def _get_least_busy_channel(self):
                # Select the least used channel
                return self.channel_usage.index(min(self.channel_usage))
                
            def get_stats(self):
                return {
                    'positions': self.channel_positions,
                    'usage': self.channel_usage,
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses + 0.001) * 100,
                    'prefetch_queue_size': len(self.prefetch_queue)
                }
            
            def prefetch(self, offset, size):
                """Prefetch data to cache"""
                # Check if already in cache
                cache_key = (offset, size)
                if cache_key in self.cache:
                    return
                
                # Check if already in prefetch queue
                with self.prefetch_lock:
                    for item in self.prefetch_queue:
                        if item[0] == offset and item[1] == size:
                            return
                    
                    # Add to prefetch queue, keep at most 10 prefetch requests
                    self.prefetch_queue.append((offset, size))
                    if len(self.prefetch_queue) > 10:
                        self.prefetch_queue.pop(0)
            
            def _prefetch_worker(self):
                """Prefetch thread"""
                while self.prefetch_thread_running:
                    try:
                        # Check prefetch queue
                        with self.prefetch_lock:
                            if self.prefetch_queue:
                                offset, size = self.prefetch_queue.pop(0)
                            else:
                                offset, size = None, None
                        
                        # If there's a prefetch request, execute prefetch
                        if offset is not None and size is not None:
                            # Check if already in cache
                            cache_key = (offset, size)
                            if cache_key not in self.cache:
                                # Choose the best channel
                                channel_id = self._get_best_channel(offset)
                                # Perform prefetch
                                self.read_chunk(offset, size, channel_id)
                    except Exception as e:
                        print(f"Prefetch error: {e}")
                    
                    # Short sleep to avoid excessive CPU usage
                    time.sleep(0.01)
                
            def close(self):
                """Close loader"""
                self.prefetch_thread_running = False
                self.cache.clear()
        
        # Create multi-channel loader
        num_channels = max(4, int(self.config['memory_channels']))  # Ensure at least 4 channels
        
        # Apply cache size from advanced settings
        cache_size = self.config.get('advanced_cache_size', 32)  # Default 32
        self.status_text.insert(tk.END, f"Memory cache size: {cache_size}\n")
        
        self.multi_channel_loader = MultiChannelLoader(
            self.mm, 
            num_channels=num_channels,
            cache_size=cache_size  # Pass cache size
        )
        
        self.status_text.insert(tk.END, f"Created {num_channels} memory channel loader with caching and prefetching\n")
        
        # Start memory monitoring thread
        self.memory_monitor_thread_running = True
        threading.Thread(target=self.memory_monitor_thread, daemon=True).start()

    def update_system_memory_stats(self):
        """Update system memory statistics"""
        try:
            # Check monitoring flag, return directly if closed
            if not self.monitoring:
                return False
                
            # Get system memory information
            mem = psutil.virtual_memory()
            
            # Update to interface
            self.status_text.insert(tk.END, f"System Memory: Used {mem.percent}% ({mem.used/1024/1024/1024:.2f}GB/{mem.total/1024/1024/1024:.2f}GB)\n")
            
            # If memory offload is enabled, monitor offload performance
            if hasattr(self, 'multi_channel_loader') and self.multi_channel_loader is not None:
                try:
                    stats = self.multi_channel_loader.get_stats()
                    channel_stats = [f"Channel {i}: {pos/1024/1024:.2f}MB" for i, pos in enumerate(stats['positions'])]
                    usage_stats = [f"Channel {i}: {usage} times" for i, usage in enumerate(stats['usage'])]
                    
                    self.status_text.insert(tk.END, f"Memory Offload Channel Status: {', '.join(channel_stats)}\n")
                    self.status_text.insert(tk.END, f"Memory Offload Channel Usage: {', '.join(usage_stats)}\n")
                    
                    # Display cache hit rate
                    if 'cache_hits' in stats and 'cache_misses' in stats:
                        total_requests = stats['cache_hits'] + stats['cache_misses']
                        if total_requests > 0:
                            hit_ratio = stats['cache_hits'] / total_requests * 100
                            self.status_text.insert(tk.END, f"Memory Cache Hit Rate: {hit_ratio:.2f}% (Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']})\n")
                except Exception as e:
                    # Capture error when getting statistics, but do not interrupt monitoring
                    self.status_text.insert(tk.END, f"Error getting memory offload statistics: {str(e)}\n")
            
            # Update GPU KV cache hit rate (if available)
            if hasattr(self, 'kv_cache_hits') and hasattr(self, 'kv_cache_misses'):
                total_kv_requests = self.kv_cache_hits + self.kv_cache_misses
                if total_kv_requests > 0:
                    kv_hit_ratio = self.kv_cache_hits / total_kv_requests * 100
                    self.status_text.insert(tk.END, f"KV Cache Hit Rate: {kv_hit_ratio:.2f}% (Hits: {self.kv_cache_hits}, Misses: {self.kv_cache_misses})\n")
            
            self.status_text.see(tk.END)
            return True
        except Exception as e:
            self.status_text.insert(tk.END, f"Memory monitoring error: {e}\n")
            return False

    def memory_monitor_thread(self):
        """Memory monitoring thread"""
        try:
            # Set local variable to avoid frequent access to self attributes
            monitoring = True
            
            while monitoring and self.monitoring:
                try:
                    if hasattr(self, 'server_process') and self.server_process is not None and self.server_process.poll() is None:
                        self.update_system_memory_stats()
                    time.sleep(5)  # Update every 5 seconds
                    
                    # Check if monitoring flag has changed
                    monitoring = self.monitoring
                except Exception as e:
                    self.status_text.insert(tk.END, f"Memory monitoring error: {str(e)}\n")
                    time.sleep(5)  # Wait 5 seconds on error before continuing
        except Exception as e:
            # Capture exception when thread starts
            print(f"Memory monitoring thread startup error: {str(e)}")

    def check_vllm_supported_args(self):
        """Check VLLM supported command line arguments"""
        supported_args = {
            'swap_space': '--swap-space',
            'cpu_offload': '--cpu-offload-gb',
            'max_cpu_memory': '--max-cpu-memory'
        }
        
        try:
            # Attempt to run vllm help command, increase timeout
            help_output = subprocess.run(
                ['vllm', 'serve', '--help'],
                capture_output=True,
                text=True,
                timeout=15  # Increase timeout to 15 seconds
            )
            
            # Check if output contains specific parameters
            output = help_output.stdout + help_output.stderr
            self.status_text.insert(tk.END, f"Checking VLLM supported parameters...\n")
            
            # Check each parameter
            if '--swap-space' not in output:
                if '--swap' in output:
                    supported_args['swap_space'] = '--swap'
                    self.status_text.insert(tk.END, "Did not find --swap-space parameter, will use --swap\n")
                else:
                    supported_args['swap_space'] = None
                    self.status_text.insert(tk.END, "No swap space related parameters found\n")
                
            # Check CPU offload parameters
            if '--cpu-offload-gb' not in output:
                if '--cpu-offload' in output:
                    supported_args['cpu_offload'] = '--cpu-offload'
                    self.status_text.insert(tk.END, "Did not find --cpu-offload-gb parameter, will use --cpu-offload\n")
                elif '--offload-params' in output:
                    supported_args['cpu_offload'] = '--offload-params'
                    self.status_text.insert(tk.END, "Did not find --cpu-offload-gb parameter, will use --offload-params\n")
                else:
                    supported_args['cpu_offload'] = None
                    self.status_text.insert(tk.END, "No CPU offload related parameters found\n")
                    
            if '--max-cpu-memory' not in output:
                supported_args['max_cpu_memory'] = None
                self.status_text.insert(tk.END, "Did not find --max-cpu-memory parameter\n")
                
            return supported_args
            
        except subprocess.TimeoutExpired:
            self.status_text.insert(tk.END, "Checking VLLM parameters timed out, using default parameters\n")
            # Use the most common parameter combination
            return {
                'swap_space': '--swap-space',
                'cpu_offload': '--cpu-offload',
                'max_cpu_memory': None
            }
        except Exception as e:
            self.status_text.insert(tk.END, f"Failed to check VLLM parameters: {str(e)}\n")
            # Return default values
            return supported_args

    def fallback_start_server(self, error_msg):
        """Fallback startup method, attempts to start server with different parameters"""
        if not messagebox.askokcancel("Error", 
            f"{error_msg}\n\nAttempt to start server using fallback method?"):
            return False
            
        self.status_text.insert(tk.END, "\nAttempting to start server using fallback method...\n")
        
        # Clean GPU memory
        self.clean_gpu_memory()
        
        # Set environment variables to avoid memory fragmentation issues
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(self.config['gpu_count'])])
        env['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
        env['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
        
        # Add VLLM specific environment variables to optimize memory usage
        env['VLLM_USE_ASYNC_CUDA_MALLOC'] = '1'  # Use asynchronous CUDA memory allocation
        env['VLLM_CPU_OFFLOAD_PIPELINE'] = '1'  # Enable CPU offload pipeline
        env['VLLM_ENABLE_STAGED_INIT'] = '1'  # Enable staged initialization
        
        self.status_text.insert(tk.END, "Optimized environment variables set\n")
        
        # Temporarily reduce model parameters
        original_max_model_len = self.config['max_model_len']
        original_max_tokens = self.config['max_tokens']
        
        # Reduce sequence length to reduce memory usage
        self.config['max_model_len'] = min(self.config['max_model_len'], 2048)  # Adjust to 2048
        self.config['max_tokens'] = min(self.config['max_tokens'], 2048)  # Adjust to 2048, ensure greater than max_num_seqs
        
        self.status_text.insert(tk.END, f"Temporarily reduced sequence length: {self.config['max_model_len']}, max tokens: {self.config['max_tokens']}\n")
        
        # Get model size
        model_size = self.estimate_model_size()
        
        # Try different startup options
        options = [
            {
                "desc": "Use minimum memory configuration",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.7',  # Reduce VRAM utilization
                    '--max-num-batched-tokens', str(self.config['max_tokens']),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',
                    '--enforce-eager'  # Add enforce-eager mode
                ]
            },
            {
                "desc": "Use quantization configuration",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.8',
                    '--max-num-batched-tokens', str(self.config['max_tokens']),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',
                    '--quantization', 'awq',  # Try AWQ quantization
                    '--enforce-eager'  # Add enforce-eager mode
                ]
            },
            {
                "desc": "Use minimum memory offload configuration",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.6',  # Further reduce VRAM utilization
                    '--max-num-batched-tokens', str(self.config['max_tokens']),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',
                    '--swap-space', '2',  # Remove GiB unit, use only number
                    '--cpu-offload-gb', '10',
                    '--enforce-eager'  # Add enforce-eager mode
                ]
            }
        ]
        
        # Add special options for large models (>10GB)
        if model_size > 10:
            # Add staged loading option
            staged_option = {
                "desc": "Use staged loading (suitable for large models)",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.5',  # Significantly reduce VRAM utilization
                    '--max-num-batched-tokens', str(min(self.config['max_tokens'], 1024)),  # Reduce batch size
                    '--block-size', str(min(self.config['block_size'], 8)),  # Reduce block size
                    '--max-model-len', str(min(self.config['max_model_len'], 1024)),  # Reduce max length
                    '--dtype', 'half',
                    '--swap-space', '4',
                    '--cpu-offload-gb', str(max(10, int(model_size * 0.7))),  # Offload at least 70% of the model
                    '--enforce-eager'  # Add enforce-eager mode
                ]
            }
            options.insert(0, staged_option)  # Place this option first
            
            # Add 8-bit quantization option
            int8_option = {
                "desc": "Use 8-bit quantization (suitable for large models)",
                "cmd": [
                    'vllm', 'serve',
                    self.config['model_path'],
                    '--host', self.config['ip'],
                    '--port', str(self.config['port']),
                    '--tensor-parallel-size', str(self.config['gpu_count']),
                    '--gpu-memory-utilization', '0.7',
                    '--max-num-batched-tokens', str(self.config['max_tokens']),
                    '--block-size', str(self.config['block_size']),
                    '--max-model-len', str(self.config['max_model_len']),
                    '--dtype', 'half',
                    '--quantization', 'int8',  # Use int8 quantization
                    '--enforce-eager'  # Add enforce-eager mode
                ]
            }
            options.insert(1, int8_option)
        
        # Try each option
        for i, option in enumerate(options):
            self.status_text.insert(tk.END, f"\nAttempting Option {i+1}: {option['desc']}\n")
            cmd_str = ' '.join(option['cmd'])
            self.status_text.insert(tk.END, f"Command: {cmd_str}\n")
                
            try:
                # Start server
                self.server_process = subprocess.Popen(
                    option['cmd'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                    
                # Wait for a short period, check if the process exits immediately
                time.sleep(5)  # Increase wait time
                if self.server_process.poll() is None:
                    # Process is still running, startup successful
                    self.status_text.insert(tk.END, "Server started successfully!\n")
                    
                    # Start monitoring thread
                    threading.Thread(target=self.monitor_server_output).start()
                    
                    # Update API address
                    api_base = f"http://{self.config['ip']}:{self.config['port']}/v1"
                    self.api_label.config(text=f"API Address: {api_base}")
                    
                    return True
                else:
                    # Process exited, get output
                    output, _ = self.server_process.communicate()
                    error_output = output.decode()
                    self.status_text.insert(tk.END, f"Startup failed: {error_output}\n")
                    
                    # Analyze error reason
                    if "CUDA out of memory" in error_output:
                        self.status_text.insert(tk.END, "Error detected: GPU out of memory\n")
                    elif "RuntimeError" in error_output:
                        self.status_text.insert(tk.END, "Error detected: Program crash\n")
                    
                    # Add extra cleanup step between options
                    self.clean_gpu_memory()
                    time.sleep(2)  # Wait for GPU memory to be released
                    
            except Exception as e:
                self.status_text.insert(tk.END, f"Attempting Option {i+1} failed: {str(e)}\n")
        
        # All options failed, provide suggestions
        self.status_text.insert(tk.END, "All fallback options failed, suggestions:\n")
        self.status_text.insert(tk.END, "1. Close other memory-intensive applications\n")
        self.status_text.insert(tk.END, "2. Restart system to clean up memory fragmentation\n")
        self.status_text.insert(tk.END, "3. Try using a quantized version of the model\n")
        self.status_text.insert(tk.END, "4. Try using a smaller model, such as 7B or smaller versions\n")
        
        # Restore original settings
        self.config['max_model_len'] = original_max_model_len
        self.config['max_tokens'] = original_max_tokens
        
        return False

    def clean_gpu_memory(self):
        """Clean GPU memory"""
        try:
            self.status_text.insert(tk.END, "Cleaning GPU memory...\n")
            
            # Attempt to free PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.status_text.insert(tk.END, "PyTorch cache cleared\n")
                
                # Get current GPU memory usage
                gpu = GPUtil.getGPUs()[0]
                free_mem = gpu.memoryFree
                total_mem = gpu.memoryTotal
                used_mem = total_mem - free_mem
                
                self.status_text.insert(tk.END, f"Current GPU memory: Used {used_mem}MB / Total {total_mem}MB\n")
                
                # If memory usage is too high, suggest user to restart system
                if used_mem / total_mem > 0.5:  # If usage exceeds 50%
                    self.status_text.insert(tk.END, "Warning: GPU memory usage is high, may affect model loading\n")
                    self.status_text.insert(tk.END, "Suggest closing other GPU-using applications or restarting the system\n")
                    
            # Attempt to run system command to free memory
            os.system("sync")  # Sync file system cache
            
            # Attempt to free system cache
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("1")
                self.status_text.insert(tk.END, "System cache cleared\n")
            except:
                pass  # May not have permissions, ignore error
                
            self.status_text.insert(tk.END, "GPU memory cleaning complete\n")
            
        except Exception as e:
            self.status_text.insert(tk.END, f"Error cleaning GPU memory: {str(e)}\n")

    def preallocate_memory_buffer(self):
        """Pre-allocate memory buffer to prevent out-of-memory during runtime"""
        try:
            self.status_text.insert(tk.END, "Pre-allocating memory buffer...\n")
                
            # Get model size
            model_size = self.estimate_model_size()
            
            # Calculate memory size to pre-allocate - dynamically adjust based on model size
            if model_size < 10:
                # Small models use smaller buffer
                buffer_size_gb = model_size * 0.2
                buffer_size_gb = max(buffer_size_gb, 4.0)  # At least 4GB
            else:
                # Large models use larger buffer, but smaller ratio
                buffer_size_gb = model_size * 0.15
                buffer_size_gb = max(buffer_size_gb, 6.0)  # At least 6GB
            
            # Check available memory
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 * 1024 * 1024)
            
            # Ensure buffer does not exceed 50% of available memory
            max_buffer_size = available_gb * 0.5
            if buffer_size_gb > max_buffer_size:
                self.status_text.insert(tk.END, f"Warning: Calculated buffer size ({buffer_size_gb:.2f}GB) exceeds 50% of available memory, adjusting size\n")
                buffer_size_gb = max_buffer_size
            
            # Reserve at least 5GB for system operation
            if available_gb < buffer_size_gb + 5:
                self.status_text.insert(tk.END, f"Warning: Insufficient available memory ({available_gb:.2f}GB), reducing buffer size\n")
                buffer_size_gb = max(2.0, available_gb - 5)  # At least 2GB, reserve 5GB for system operation
                
            self.status_text.insert(tk.END, f"Pre-allocating memory buffer size: {buffer_size_gb:.2f}GB\n")
            
            # Create memory buffer directory
            buffer_dir = os.path.join(os.getcwd(), "memory_buffer")
            os.makedirs(buffer_dir, exist_ok=True)
            
            # Create memory buffer file
            buffer_file = os.path.join(buffer_dir, "memory_buffer.bin")
            
            # If file already exists, check if size is sufficient
            if os.path.exists(buffer_file):
                current_size = os.path.getsize(buffer_file) / (1024 * 1024 * 1024)
                if current_size >= buffer_size_gb:
                    self.status_text.insert(tk.END, f"Using existing memory buffer: {current_size:.2f}GB\n")
                    return
                else:
                    self.status_text.insert(tk.END, f"Existing memory buffer size is insufficient ({current_size:.2f}GB), recreating\n")
                    os.remove(buffer_file)
            
            # Create new memory buffer file
            self.status_text.insert(tk.END, f"Creating memory buffer file: {buffer_file}\n")
            
            # Calculate buffer size (bytes)
            buffer_size_bytes = int(buffer_size_gb * 1024 * 1024 * 1024)
            
            # Create memory buffer file
            with open(buffer_file, "wb") as f:
                # Write in chunks to avoid allocating too much memory at once
                chunk_size = 1024 * 1024 * 64  # Reduce to 64MB chunks to lower memory pressure
                remaining = buffer_size_bytes
                
                # Log memory usage
                mem_before = psutil.virtual_memory()
                self.status_text.insert(tk.END, f"System memory before creation: Used {mem_before.percent}% ({mem_before.used/1024/1024/1024:.2f}GB/{mem_before.total/1024/1024/1024:.2f}GB)\n")
                
                try:
                    while remaining > 0:
                        # Check memory status every 256MB written, more frequent checks
                        if (buffer_size_bytes - remaining) % (256*1024*1024) < chunk_size:
                            mem_check = psutil.virtual_memory()
                            # If available memory is below 2.5GB, stop writing
                            if mem_check.available < 2.5 * 1024 * 1024 * 1024:
                                self.status_text.insert(tk.END, f"Warning: Available memory below 2.5GB, stopping further memory allocation\n")
                                break
                        
                        write_size = min(chunk_size, remaining)
                        f.write(b'\0' * write_size)
                        remaining -= write_size
                        # Update progress
                        progress = (buffer_size_bytes - remaining) / buffer_size_bytes * 100
                        self.status_text.delete("end-2l", "end-1l")  # Delete previous progress line
                        self.status_text.insert(tk.END, f"Creating memory buffer: {progress:.1f}% ({(buffer_size_bytes-remaining)/(1024*1024*1024):.2f}GB/{buffer_size_gb:.2f}GB)\n")
                        self.status_text.see(tk.END)
                except MemoryError:
                    self.status_text.insert(tk.END, f"Out of memory, cannot complete buffer creation\n")
                    # Record allocated size
                    actual_size = buffer_size_bytes - remaining
                    self.status_text.insert(tk.END, f"Allocated {actual_size/(1024*1024*1024):.2f}GB\n")
                    # Truncate file to written size
                    f.flush()
                    f.truncate(actual_size)
                
                # Log memory usage
                mem_after = psutil.virtual_memory()
                self.status_text.insert(tk.END, f"System memory after creation: Used {mem_after.percent}% ({mem_after.used/1024/1024/1024:.2f}GB/{mem_after.total/1024/1024/1024:.2f}GB)\n")
        
            # Verify final file size
            final_size = os.path.getsize(buffer_file)
            self.status_text.insert(tk.END, f"Memory buffer final size: {final_size/(1024*1024*1024):.2f}GB\n")
            
            # Open file and map to memory
            self.buffer_file = open(buffer_file, "r+b")
            self.buffer_mm = mmap.mmap(self.buffer_file.fileno(), 0)
            
            self.status_text.insert(tk.END, f"Memory buffer creation complete: {final_size/(1024*1024*1024):.2f}GB\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Error creating memory buffer: {str(e)}\n")
            import traceback
            self.status_text.insert(tk.END, traceback.format_exc())
    
    def cleanup_memory_buffer(self):
        """Clean up memory buffer"""
        try:
            if hasattr(self, 'buffer_mm') and self.buffer_mm:
                self.buffer_mm.close()
                self.buffer_mm = None
            
            if hasattr(self, 'buffer_file') and self.buffer_file:
                self.buffer_file.close()
                self.buffer_file = None
            
            self.status_text.insert(tk.END, "Memory buffer released\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Error releasing memory buffer: {str(e)}\n")

    def recommend_settings(self):
        """Recommend settings based on model size and hardware conditions"""
        try:
            # Check if model is selected
            if not self.config['model_path']:
                messagebox.showerror("Error", "Please select a model path first")
                return
                
            # Estimate model size
            model_size = self.estimate_model_size()
            
            # Get GPU information
            gpus = GPUtil.getGPUs()
            if not gpus:
                messagebox.showerror("Error", "No GPU detected")
                return
                
            # Get VRAM size of the first GPU (GB)
            gpu_memory = gpus[0].memoryTotal / 1024
            
            # Get system memory size (GB)
            system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            # Recommend settings based on model size and hardware conditions
            self.status_text.insert(tk.END, "\n===== Recommended Settings =====\n")
            self.status_text.insert(tk.END, f"Model Size: {model_size:.2f}GB\n")
            self.status_text.insert(tk.END, f"GPU VRAM: {gpu_memory:.2f}GB\n")
            self.status_text.insert(tk.END, f"System Memory: {system_memory:.2f}GB\n")
            
            # Recommended VRAM ratio
            if model_size > gpu_memory * 0.9:
                # Model size close to or exceeding VRAM, memory offload needed
                mem_ratio = 85  # Reduce to 85% to leave more margin for the system
                self.status_text.insert(tk.END, f"Recommended VRAM Ratio: {mem_ratio}% (Large model, reduce ratio to avoid OOM)\n")
                
                # Enable memory offload
                self.enable_offload_var.set(True)
                
                # Calculate reasonable memory offload ratio
                if model_size > gpu_memory * 1.5:
                    # Model much larger than VRAM, large offload needed
                    offload_ratio = 70  # Reduce to 70% to avoid excessive system memory pressure
                else:
                    # Model slightly larger than VRAM, moderate offload
                    offload_ratio = 60
                    
                self.memory_offload_ratio_var.set(str(offload_ratio))
                self.status_text.insert(tk.END, f"Recommended Memory Offload Ratio: {offload_ratio}%\n")
                
                # Recommended memory channel count - adjust based on system memory size
                if system_memory > 64:  # Only recommend more channels for large memory systems
                    channels = 8
                else:
                    channels = 4  # For 32GB memory systems, use 4 channels
                    
                self.memory_channels_var.set(str(channels))
                self.status_text.insert(tk.END, f"Recommended Memory Channels: {channels}\n")
                
                # Recommended reserved memory ratio
                reserved_memory = 20
                self.reserved_memory_var.set(str(reserved_memory))
                self.status_text.insert(tk.END, f"Recommended Reserved System Memory: {reserved_memory}%\n")
                
                # Recommend smaller sequence length
                if model_size > 20:
                    max_model_len = 2048
                else:
                    max_model_len = 4096
                    
                self.max_model_len_var.set(str(max_model_len))
                self.status_text.insert(tk.END, f"Recommended Max Sequence Length: {max_model_len}\n")
                
                # Recommend moderate block size to improve memory bandwidth utilization
                block_size = 32  # For general hardware, 32 is a good balance
                self.block_size_var.set(str(block_size))
                self.status_text.insert(tk.END, f"Recommended Block Size: {block_size} (Improves memory bandwidth utilization)\n")
                
                # Recommend using --enforce-eager parameter
                self.status_text.insert(tk.END, "Recommend using enforce-eager mode to avoid out-of-memory during CUDA graph capture phase\n")
                
            else:
                # Model can fit entirely in VRAM
                mem_ratio = 90
                self.status_text.insert(tk.END, f"Recommended VRAM Ratio: {mem_ratio}% (Model can fit entirely in VRAM)\n")
                
                # No memory offload needed
                self.enable_offload_var.set(False)
                self.status_text.insert(tk.END, "No memory offload needed\n")
                
                # Recommend larger sequence length
                max_model_len = 8192
                self.max_model_len_var.set(str(max_model_len))
                self.status_text.insert(tk.END, f"Recommended Max Sequence Length: {max_model_len}\n")
                
                # Recommend moderate block size to improve memory bandwidth utilization
                block_size = 32  # For general hardware, 32 is a good balance
                self.block_size_var.set(str(block_size))
                self.status_text.insert(tk.END, f"Recommended Block Size: {block_size} (Improves memory bandwidth utilization)\n")
            
            # Update values on the interface
            self.mem_ratio_entry.delete(0, tk.END)
            self.mem_ratio_entry.insert(0, str(mem_ratio))
            
            # Update configuration
            self.update_config()
            
            self.status_text.insert(tk.END, "Recommended settings applied to interface\n")
            self.status_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to recommend settings: {str(e)}")

    def update_config(self):
        """Update configuration parameters"""
        try:
            # Get values from the interface
            model_path = self.model_path_entry.get()
            ip = self.ip_entry.get()
            port = int(self.port_entry.get())
            gpu_count = int(self.gpu_count_var.get())
            mem_ratio = int(self.mem_ratio_entry.get())
            max_tokens = int(self.max_tokens_var.get())
            max_model_len = int(self.max_model_len_var.get())
            block_size = int(self.block_size_var.get())
            
            # Get memory offload configuration
            enable_memory_offload = self.enable_offload_var.get()
            memory_channels = int(self.memory_channels_var.get())
            memory_offload_ratio = int(self.memory_offload_ratio_var.get())
            reserved_memory = int(self.reserved_memory_var.get())
            
            # Validate parameters
            if port < 1 or port > 65535:
                messagebox.showerror("Error", "Port number must be between 1 and 65535")
                return False
                
            if gpu_count < 1:
                messagebox.showerror("Error", "GPU count must be greater than 0")
                return False
                
            if mem_ratio < 10 or mem_ratio > 100:
                messagebox.showerror("Error", "VRAM ratio must be between 10 and 100")
                return False
                
            if max_tokens < 256:
                messagebox.showerror("Error", "Max tokens cannot be less than 256")
                return False
                
            if max_model_len < 512:
                messagebox.showerror("Error", "Max model length cannot be less than 512")
                return False
                
            if block_size < 1:
                messagebox.showerror("Error", "Block size must be greater than 0")
                return False
                
            # Validate memory offload configuration
            if enable_memory_offload:
                if memory_channels < 1:
                    messagebox.showerror("Error", "Memory channel count must be greater than 0")
                    return False
                    
                if memory_offload_ratio < 10 or memory_offload_ratio > 100:
                    messagebox.showerror("Error", "Memory offload ratio must be between 10 and 100")
                    return False
                    
                if reserved_memory < 0 or reserved_memory > 50:
                    messagebox.showerror("Error", "Reserved memory ratio must be between 0 and 50")
                    return False
            
            # Update configuration
            self.config['model_path'] = model_path
            self.config['ip'] = ip
            self.config['port'] = port
            self.config['gpu_count'] = gpu_count
            self.config['mem_ratio'] = mem_ratio
            self.config['max_tokens'] = max_tokens
            self.config['max_model_len'] = max_model_len
            self.config['block_size'] = block_size
            
            # Update memory offload configuration
            self.config['enable_memory_offload'] = enable_memory_offload
            self.config['memory_channels'] = memory_channels
            self.config['memory_offload_ratio'] = memory_offload_ratio
            self.config['reserved_memory'] = reserved_memory
            
            # Save configuration to file
            self.save_config()
            
            # Display configuration information in status bar
            self.status_text.insert(tk.END, "\n===== Configuration Updated =====\n")
            self.status_text.insert(tk.END, f"Model Path: {model_path}\n")
            self.status_text.insert(tk.END, f"IP Address: {ip}, Port: {port}\n")
            self.status_text.insert(tk.END, f"GPU Count: {gpu_count}, VRAM Ratio: {mem_ratio}%\n")
            self.status_text.insert(tk.END, f"Max Tokens: {max_tokens}, Max Model Length: {max_model_len}, Block Size: {block_size}\n")
            
            if enable_memory_offload:
                self.status_text.insert(tk.END, f"Memory offload enabled: Channels={memory_channels}, Offload Ratio={memory_offload_ratio}%, Reserved Memory={reserved_memory}%\n")
            else:
                self.status_text.insert(tk.END, "Memory offload not enabled\n")
                
            self.status_text.see(tk.END)
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update configuration: {str(e)}")
            return False

    def validate_config(self):
        """Validate configuration parameters"""
        if self.config['max_tokens'] < self.config['max_model_len']:
            if not messagebox.askokcancel("Warning", 
                "Max response tokens is less than total sequence length, this may affect model performance.\nSuggest setting max_tokens to be not less than max_model_len.\nContinue?"):
                return False
        return True

    def check_model_compatibility(self):
        """Check model compatibility with VLLM"""
        if not self.config['model_path']:
            self.status_text.insert(tk.END, "Error: Model path not selected\n")
            return False
        
        self.status_text.insert(tk.END, "Checking model compatibility...\n")
        
        # Check hardware configuration
        self.check_hardware_configuration()
        
        # Check if model files exist
        model_path = self.config['model_path']
        if not os.path.exists(model_path):
            self.status_text.insert(tk.END, f"Error: Model path does not exist: {model_path}\n")
            return False
        
        # Check necessary model files
        required_files = []
        safetensors_found = False
        bin_files_found = False
        
        # Check for .safetensors files
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.safetensors'):
                    safetensors_found = True
                    self.status_text.insert(tk.END, f"Found safetensors file: {file}\n")
                elif file.endswith('.bin'):
                    bin_files_found = True
                    self.status_text.insert(tk.END, f"Found bin file: {file}\n")
        
        if not (safetensors_found or bin_files_found):
            self.status_text.insert(tk.END, "Error: Model weight file (.safetensors or .bin) not found\n")
            return False
        
        # Check config.json file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            self.status_text.insert(tk.END, "Error: config.json file not found\n")
            return False
        
        # Check tokenizer files
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
        tokenizer_found = False
        for file in tokenizer_files:
            if os.path.exists(os.path.join(model_path, file)):
                tokenizer_found = True
                break
        
        if not tokenizer_found:
            self.status_text.insert(tk.END, "Warning: Standard tokenizer file not found, VLLM may not load correctly\n")
        
        # Read model configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check model type
            model_type = config.get('model_type', '')
            self.status_text.insert(tk.END, f"Model Type: {model_type}\n")
            
            # Check if it's a supported model type
            supported_types = ["llama", "mistral", "falcon", "gpt_neox", "gpt2", "bloom", "qwen", "baichuan", "chatglm", "mpt"]
            if model_type.lower() not in [t.lower() for t in supported_types]:
                self.status_text.insert(tk.END, f"Warning: Model type '{model_type}' may not be fully supported by VLLM\n")
            
            # Check model size
            hidden_size = config.get('hidden_size', 0)
            num_layers = config.get('num_hidden_layers', 0) or config.get('num_layers', 0)
            vocab_size = config.get('vocab_size', 0)
            
            if hidden_size and num_layers:
                # Roughly estimate model parameters
                params_billion = (hidden_size * hidden_size * 4 * num_layers + hidden_size * vocab_size) / 1e9
                self.status_text.insert(tk.END, f"Estimated Model Parameters: {params_billion:.2f}B\n")
                
                # Check if it's a large model
                if params_billion > 30:
                    self.status_text.insert(tk.END, "Warning: This is a large model, may require multi-GPU or memory offload\n")
            
            # Check special attention mechanism
            attention_type = config.get('attention_type', '')
            if attention_type and attention_type not in ['scaled_dot_product', 'eager']:
                self.status_text.insert(tk.END, f"Warning: Special attention mechanism '{attention_type}' may not be supported by VLLM\n")
            
            # Check activation function
            activation_function = config.get('hidden_act', '')
            if activation_function and activation_function not in ['gelu', 'gelu_new', 'relu', 'silu', 'swish']:
                self.status_text.insert(tk.END, f"Warning: Activation function '{activation_function}' may not be fully supported by VLLM\n")
        
        except Exception as e:
            self.status_text.insert(tk.END, f"Error reading model configuration: {str(e)}\n")
        
        # Check VLLM version
        try:
            vllm_version = subprocess.run(['vllm', '--version'], capture_output=True, text=True)
            version_str = vllm_version.stdout.strip() or vllm_version.stderr.strip()
            self.status_text.insert(tk.END, f"VLLM Version: {version_str}\n")
            
            # Check CUDA version
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                self.status_text.insert(tk.END, f"CUDA Version: {cuda_version}\n")
                
                # Check GPU compute capability
                capability = torch.cuda.get_device_capability()
                self.status_text.insert(tk.END, f"GPU Compute Capability: {capability[0]}.{capability[1]}\n")
                
                # Check if current GPU is supported
                if capability[0] < 7:
                    self.status_text.insert(tk.END, "Warning: VLLM best supports GPUs with compute capability 7.0+ (V100 and newer)\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Error checking VLLM version: {str(e)}\n")
        
        # Check GPU memory
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_memory = gpu.memoryTotal / 1024  # GB
                self.status_text.insert(tk.END, f"GPU VRAM: {gpu_memory:.2f}GB\n")
                
                # Estimate model size
                model_size = self.estimate_model_size()
                self.status_text.insert(tk.END, f"Estimated Model Size: {model_size:.2f}GB\n")
                
                # Check if memory offload is needed
                if model_size > gpu_memory * 0.8:
                    self.status_text.insert(tk.END, f"Warning: Model size ({model_size:.2f}GB) is close to or exceeds GPU VRAM ({gpu_memory:.2f}GB)\n")
                    self.status_text.insert(tk.END, "Suggest enabling memory offload or using multi-GPU\n")
                    
                    # Check system memory
                    system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
                    self.status_text.insert(tk.END, f"System Memory: {system_memory:.2f}GB\n")
                    
                    if system_memory < model_size * 1.5:
                        self.status_text.insert(tk.END, "Warning: System memory may be insufficient for effective memory offload\n")
                    
                    # Check disk space (for memory-mapped file)
                    disk_usage = psutil.disk_usage('/')
                    free_disk = disk_usage.free / (1024 * 1024 * 1024)  # GB
                    self.status_text.insert(tk.END, f"Available Disk Space: {free_disk:.2f}GB\n")
                    
                    if free_disk < model_size * 2:
                        self.status_text.insert(tk.END, "Warning: Disk space may be insufficient to create memory-mapped file\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Error checking GPU memory: {str(e)}\n")
        
        self.status_text.insert(tk.END, "Model compatibility check complete\n")
        return True

    def check_hardware_configuration(self):
        """Detect user hardware configuration and provide corresponding optimization suggestions"""
        self.status_text.insert(tk.END, "\n===== Hardware Configuration Detection =====\n")
        
        # Detect CPU
        try:
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            cpu_logical = psutil.cpu_count(logical=True)  # Logical cores
            self.status_text.insert(tk.END, f"CPU: {cpu_count} Cores/{cpu_logical} Threads\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Unable to detect CPU information: {str(e)}\n")
        
        # Detect Memory
        try:
            mem = psutil.virtual_memory()
            total_memory = mem.total / (1024 * 1024 * 1024)  # GB
            self.status_text.insert(tk.END, f"System Memory: {total_memory:.2f}GB\n")
            
            # Provide suggestions based on memory size
            if total_memory < 16:
                self.status_text.insert(tk.END, "Warning: System memory is low, may not be able to run large models\n")
                self.status_text.insert(tk.END, "Suggestion: Use smaller models or quantized models, reduce block_size and max_model_len\n")
            elif total_memory < 32:
                self.status_text.insert(tk.END, "System memory is moderate, can run medium-sized models\n")
                self.status_text.insert(tk.END, "Suggestion: Use default settings, or adjust appropriately based on model size\n")
            else:
                self.status_text.insert(tk.END, "System memory is sufficient, can run larger models\n")
                self.status_text.insert(tk.END, "Suggestion: You can appropriately increase memory channels and cache size\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Unable to detect memory information: {str(e)}\n")
        
        # Detect GPU
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.status_text.insert(tk.END, f"Detected {gpu_count} GPUs\n")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024 * 1024)  # GB
                    self.status_text.insert(tk.END, f"GPU {i}: {gpu_name}, VRAM: {gpu_mem:.2f}GB\n")
                    
                    # Provide suggestions based on VRAM size
                    if gpu_mem < 8:
                        self.status_text.insert(tk.END, f"Warning: GPU {i} VRAM is low, may not be able to run large models\n")
                        self.status_text.insert(tk.END, f"Suggestion: Use smaller models or quantized models, reduce block_size\n")
                    elif gpu_mem < 16:
                        self.status_text.insert(tk.END, f"GPU {i} VRAM is moderate, can run medium-sized models\n")
                        self.status_text.insert(tk.END, f"Suggestion: Use default settings, block_size set to 16-32\n")
                    else:
                        self.status_text.insert(tk.END, f"GPU {i} VRAM is sufficient, can run larger models\n")
                        self.status_text.insert(tk.END, f"Suggestion: You can increase block_size to 32-64\n")
            else:
                self.status_text.insert(tk.END, "No CUDA-enabled GPU detected\n")
                self.status_text.insert(tk.END, "Warning: VLLM requires NVIDIA GPU to run\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Error detecting GPU information: {str(e)}\n")
        
        # Detect physical memory channel count (this is an estimate, cannot be accurately obtained)
        try:
            # Attempt to get memory channel information via dmidecode (Linux only, requires root privileges)
            if sys.platform.startswith('linux'):
                try:
                    result = subprocess.run(['sudo', 'dmidecode', '-t', 'memory'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        # Analyze output, estimate memory channel count
                        memory_devices = result.stdout.count("Memory Device")
                        if memory_devices > 0:
                            self.status_text.insert(tk.END, f"Estimated physical memory channels: {memory_devices}\n")
                            
                            # Provide suggestions based on memory channel count
                            if memory_devices <= 2:
                                self.status_text.insert(tk.END, "Physical memory channels are few, may limit memory bandwidth\n")
                                self.status_text.insert(tk.END, "Suggestion: Use 4-8 software memory channels\n")
                            else:
                                self.status_text.insert(tk.END, "Physical memory channels are sufficient\n")
                                self.status_text.insert(tk.END, "Suggestion: You can use 8-16 software memory channels\n")
                except Exception as e:
                    # If unable to get, use default estimate
                    self.status_text.insert(tk.END, "Unable to accurately detect physical memory channels, using default estimate\n")
                    self.status_text.insert(tk.END, "Suggestion: Use 4-8 software memory channels, avoid setting too high\n")
            else:
                # Non-Linux system
                self.status_text.insert(tk.END, "Unable to detect physical memory channels on current operating system\n")
                self.status_text.insert(tk.END, "Suggestion: Use 4-8 software memory channels, avoid setting too high\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Unable to detect physical memory channels: {str(e)}\n")
            self.status_text.insert(tk.END, "Suggestion: Use default settings (4 software memory channels)\n")
        
        self.status_text.insert(tk.END, "Hardware configuration detection complete\n")
        self.status_text.see(tk.END)

    def check_flash_attention_support(self):
        """Check for Flash Attention support"""
        try:
            import torch
            has_support = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            self.status_text.insert(tk.END, f"Flash Attention Support: {has_support}\n")
            return False  # Temporarily disable Flash Attention feature to avoid compatibility issues
        except Exception as e:
            self.status_text.insert(tk.END, f"Error checking Flash Attention support: {str(e)}\n")
            return False

    def add_performance_monitoring(self):
        """Add performance monitoring and auto-tuning features"""
        # Create performance monitoring panel
        self.perf_frame = ttk.LabelFrame(self.master, text="Performance Monitoring")
        self.perf_frame.pack(padx=10, pady=5, fill='both')
        
        # Add performance metric display
        self.perf_labels = {}
        metrics = ["GPU Utilization", "Memory Bandwidth", "KV Cache Hit Rate", "Inference Speed (token/s)"]
        
        for i, metric in enumerate(metrics):
            ttk.Label(self.perf_frame, text=f"{metric}:").grid(row=i, column=0, sticky='w')
            self.perf_labels[metric] = ttk.Label(self.perf_frame, text="N/A")
            self.perf_labels[metric].grid(row=i, column=1, sticky='w')
        
        # Add auto-tuning switch
        self.auto_tune_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.perf_frame, text="Enable Automatic Performance Tuning", variable=self.auto_tune_var).grid(row=len(metrics), column=0, columnspan=2, sticky='w')
        
        # Initialize performance statistics variables
        self.total_tokens_generated = 0
        self.kv_cache_hits = 0
        self.kv_cache_misses = 0
        
        # Start performance monitoring thread
        self.start_performance_monitor()

    def start_performance_monitor(self):
        """Start performance monitoring thread"""
        def monitor_loop():
            last_tokens = 0
            last_time = time.time()
            
            while hasattr(self, 'monitoring') and self.monitoring:
                try:
                    if hasattr(self, 'server_process') and self.server_process is not None and self.server_process.poll() is None:
                        # Get GPU statistics
                        gpu_stats = self.get_gpu_stats()
                        if gpu_stats and len(gpu_stats) > 0:
                            # Safely get GPU utilization and memory utilization
                            gpu_util_str = gpu_stats[0].get('utilization.gpu', '0 %').replace('%', '').strip()
                            mem_util_str = gpu_stats[0].get('utilization.memory', '0 %').replace('%', '').strip()
                            
                            # Convert to float, handle possible conversion errors
                            try:
                                gpu_util = float(gpu_util_str)
                            except ValueError:
                                gpu_util = 0.0
                                
                            try:
                                mem_util = float(mem_util_str)
                            except ValueError:
                                mem_util = 0.0
                            
                            # Update GUI display
                            if 'GPU Utilization' in self.perf_labels:
                                self.perf_labels['GPU Utilization'].config(text=f"{gpu_util:.1f}%")
                            if 'Memory Bandwidth' in self.perf_labels:
                                self.perf_labels['Memory Bandwidth'].config(text=f"{mem_util:.1f}%")
                            
                            # Calculate inference speed
                            if hasattr(self, 'total_tokens_generated'):
                                now = time.time()
                                if now - last_time >= 5:  # Update every 5 seconds
                                    tokens_per_sec = (self.total_tokens_generated - last_tokens) / (now - last_time)
                                    last_tokens = self.total_tokens_generated
                                    last_time = now
                                    
                                    if 'Inference Speed (token/s)' in self.perf_labels:
                                        self.perf_labels['Inference Speed (token/s)'].config(text=f"{tokens_per_sec:.2f}")
                                    
                                    # Auto-tuning logic - only execute when service is running and auto-tuning is enabled
                                    if hasattr(self, 'monitoring') and self.monitoring and hasattr(self, 'auto_tune_var') and self.auto_tune_var.get() and tokens_per_sec < 5.0:
                                        # If GPU utilization is high but memory bandwidth is low, it indicates a memory bottleneck
                                        if gpu_util > 90 and mem_util < 30:
                                            self.status_text.insert(tk.END, "GPU compute bottleneck detected, attempting to optimize memory access...\n")
                                            self.optimize_memory_access()
                                        # If GPU utilization is low, it indicates a compute bottleneck
                                        elif gpu_util < 30:
                                            self.status_text.insert(tk.END, "Low GPU utilization detected, attempting auto-tuning...\n")
                                            self.optimize_for_low_gpu_utilization()
                
                    # Update KV Cache Hit Rate
                    if hasattr(self, 'monitoring') and self.monitoring and hasattr(self, 'kv_cache_hits') and hasattr(self, 'kv_cache_misses'):
                        total_kv_requests = self.kv_cache_hits + self.kv_cache_misses
                        if total_kv_requests > 0:
                            kv_hit_ratio = self.kv_cache_hits / total_kv_requests * 100
                            if 'KV Cache Hit Rate' in self.perf_labels:
                                self.perf_labels['KV Cache Hit Rate'].config(text=f"{kv_hit_ratio:.2f}%")
                except Exception as e:
                    # Log error but do not interrupt monitoring
                    print(f"Performance monitoring error: {str(e)}")
                
                # Check monitoring flag
                if not hasattr(self, 'monitoring') or not self.monitoring:
                    break
                
                time.sleep(1)
        
        # Ensure monitoring attribute is set
        if not hasattr(self, 'monitoring'):
            self.monitoring = True
            
        # Start monitoring thread
        self.perf_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.perf_monitor_thread.start()

    def optimize_for_low_gpu_utilization(self):
        """Optimize for low GPU utilization"""
        # This method will be called when GPU utilization is below 30%
        
        # 1. Attempt to increase batch size
        if hasattr(self, 'batch_size'):
            old_batch_size = self.batch_size
            self.batch_size = min(self.batch_size * 2, 32)  # Max batch size 32
            if old_batch_size != self.batch_size:
                self.status_text.insert(tk.END, f"Auto-tuning: Increasing batch size to {self.batch_size}\n")
        
        # 2. Attempt to warm up GPU
        self.status_text.insert(tk.END, "Auto-tuning: Performing GPU warm-up operation\n")
        try:
            # Create a small tensor and perform some operations to warm up the GPU
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Create a large tensor and perform some operations
                x = torch.randn(1000, 1000, device=device)
                for _ in range(10):
                    x = torch.matmul(x, x)
                # Force synchronization
                torch.cuda.synchronize()
                self.status_text.insert(tk.END, "GPU warm-up complete\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"GPU warm-up failed: {str(e)}\n")
        
        # 3. Check and optimize memory access patterns
        if hasattr(self, 'multi_channel_loader'):
            # Increase cache size
            if hasattr(self.multi_channel_loader, 'max_cache_size'):
                old_cache_size = self.multi_channel_loader.max_cache_size
                self.multi_channel_loader.max_cache_size = min(old_cache_size * 2, 128)
                self.status_text.insert(tk.END, f"Auto-tuning: Increasing memory cache size to {self.multi_channel_loader.max_cache_size}\n")

    def optimize_memory_access(self):
        """Optimize memory access patterns to improve memory bandwidth utilization"""
        self.status_text.insert(tk.END, "Optimizing memory access patterns...\n")
        
        # 1. Increase prefetching operations
        if hasattr(self, 'multi_channel_loader') and self.multi_channel_loader is not None:
            try:
                # Increase cache size, but not too large
                if hasattr(self.multi_channel_loader, 'max_cache_size'):
                    old_cache_size = self.multi_channel_loader.max_cache_size
                    # For general hardware, max increase to 64
                    self.multi_channel_loader.max_cache_size = min(old_cache_size * 2, 64)
                    self.status_text.insert(tk.END, f"Increasing memory cache size to {self.multi_channel_loader.max_cache_size}\n")
            except Exception as e:
                self.status_text.insert(tk.END, f"Failed to optimize memory cache: {str(e)}\n")
        
        # 2. Attempt to optimize CUDA memory allocation strategy
        try:
            # Set environment variables to optimize CUDA memory allocation, but use smaller chunk size
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
            self.status_text.insert(tk.END, "CUDA memory allocation strategy optimized\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"Failed to optimize CUDA memory allocation: {str(e)}\n")
        
        # 3. Attempt to adjust KV cache parameters
        if hasattr(self, 'server_process') and self.server_process is not None and self.server_process.poll() is None:
            self.status_text.insert(tk.END, "Suggest restarting server and using a more appropriate block_size to improve memory bandwidth utilization\n")
            
            # Update recommended settings
            current_block_size = int(self.block_size_var.get())
            # For general hardware, max recommended is 32
            recommended_block_size = min(current_block_size * 2, 32)
            self.status_text.insert(tk.END, f"Recommended block_size: {recommended_block_size} (Current: {current_block_size})\n")

    def update_token_count(self, new_tokens):
        """Update generated token count"""
        if not hasattr(self, 'total_tokens_generated'):
            self.total_tokens_generated = 0
        self.total_tokens_generated += new_tokens

    def auto_optimize_performance(self):
        """Automatically perform performance optimization after server startup"""
        try:
            # Wait for server to fully start
            time.sleep(10)
            
            if not self.monitoring or not hasattr(self, 'server_process') or self.server_process is None or self.server_process.poll() is not None:
                return
                
            self.status_text.insert(tk.END, "\n===== Automatic Performance Optimization =====\n")
            
            # 1. Warm up GPU
            self.status_text.insert(tk.END, "Performing GPU warm-up...\n")
            self.warm_up_gpu()
            
            # 2. Optimize memory access patterns
            self.status_text.insert(tk.END, "Optimizing memory access patterns...\n")
            self.optimize_memory_access()
            
            # 3. Provide performance optimization suggestions
            self.status_text.insert(tk.END, "Performance Optimization Suggestions:\n")
            
            # Check block size
            current_block_size = int(self.block_size_var.get())
            if current_block_size < 32:
                self.status_text.insert(tk.END, f"- Current block size ({current_block_size}) is small, suggest using a block size of 32 for next startup to improve memory bandwidth utilization\n")
            
            # Check memory channel count
            if hasattr(self, 'multi_channel_loader') and self.multi_channel_loader is not None:
                current_channels = self.multi_channel_loader.num_channels
                if current_channels < 4:
                    self.status_text.insert(tk.END, f"- Current memory channel count ({current_channels}) is low, suggest using 4-8 channels for next startup to improve memory bandwidth utilization\n")
            
            # Check KV cache settings
            self.status_text.insert(tk.END, "- Using a larger batch size can increase throughput, but will increase latency\n")
            self.status_text.insert(tk.END, "- If memory bandwidth utilization is still low, you can try using a quantized model\n")
            
            # Add hardware upgrade suggestions
            self.status_text.insert(tk.END, "Hardware Upgrade Suggestions:\n")
            self.status_text.insert(tk.END, "- If you have more memory (>64GB) and more physical memory channels (>2), you can manually increase memory channels and cache size in the interface\n")
            self.status_text.insert(tk.END, "- If you have a GPU with larger VRAM (>16GB), you can try increasing block size to 64\n")
            
            self.status_text.insert(tk.END, "Automatic performance optimization complete\n")
            self.status_text.see(tk.END)
            
        except Exception as e:
            self.status_text.insert(tk.END, f"Automatic performance optimization failed: {str(e)}\n")

    def warm_up_gpu(self):
        """Warm up GPU to improve performance stability"""
        try:
            # Create a small tensor and perform some operations to warm up the GPU
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Create a large tensor and perform some operations
                x = torch.randn(2000, 2000, device=device)
                for _ in range(20):
                    x = torch.matmul(x, x)
                # Force synchronization
                torch.cuda.synchronize()
                self.status_text.insert(tk.END, "GPU warm-up complete\n")
        except Exception as e:
            self.status_text.insert(tk.END, f"GPU warm-up failed: {str(e)}\n")

    def create_advanced_settings(self):
        """Create advanced performance settings area"""
        # Create advanced settings frame
        advanced_frame = ttk.LabelFrame(self.master, text="Advanced Performance Settings")
        advanced_frame.pack(padx=10, pady=5, fill='x')
        
        # Add description
        ttk.Label(advanced_frame, text="The following settings are for high-performance hardware, please adjust carefully based on your actual hardware configuration", 
                 foreground="red").grid(row=0, column=0, columnspan=4, sticky='w')
        
        # Memory Cache Size
        ttk.Label(advanced_frame, text="Memory Cache Size:").grid(row=1, column=0)
        self.cache_size_var = tk.StringVar(value="32")
        cache_size_combo = ttk.Combobox(advanced_frame, textvariable=self.cache_size_var,
                                      values=["16", "32", "64", "128", "256"], width=5)
        cache_size_combo.grid(row=1, column=1)
        ttk.Label(advanced_frame, text="(Can be increased for large memory systems)").grid(row=1, column=2)
        
        # CUDA Memory Allocation Block Size
        ttk.Label(advanced_frame, text="CUDA Memory Chunk (MB):").grid(row=2, column=0)
        self.cuda_split_size_var = tk.StringVar(value="128")
        cuda_split_combo = ttk.Combobox(advanced_frame, textvariable=self.cuda_split_size_var,
                                      values=["64", "128", "256", "512"], width=5)
        cuda_split_combo.grid(row=2, column=1)
        ttk.Label(advanced_frame, text="(Can be increased for large VRAM GPUs)").grid(row=2, column=2)
        
        # Batch Size
        ttk.Label(advanced_frame, text="Batch Size:").grid(row=3, column=0)
        self.batch_size_var = tk.StringVar(value="16")
        batch_size_combo = ttk.Combobox(advanced_frame, textvariable=self.batch_size_var,
                                      values=["8", "16", "32", "64"], width=5)
        batch_size_combo.grid(row=3, column=1)
        ttk.Label(advanced_frame, text="(Can be increased for high-performance GPUs)").grid(row=3, column=2)
        
        # Detect Hardware Button
        detect_hardware_button = ttk.Button(advanced_frame, text="Detect Hardware Configuration", 
                                          command=self.check_hardware_configuration)
        detect_hardware_button.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Apply Advanced Settings Button
        apply_advanced_button = ttk.Button(advanced_frame, text="Apply Advanced Settings", 
                                         command=self.apply_advanced_settings)
        apply_advanced_button.grid(row=4, column=2, columnspan=2, pady=5)
        
        # Add description
        ttk.Label(advanced_frame, text="Note: Advanced settings will take effect on next server startup", 
                 foreground="blue").grid(row=5, column=0, columnspan=4, sticky='w')
        
        # Load saved advanced settings
        self.load_advanced_settings()

    def load_advanced_settings(self):
        """Load saved advanced settings"""
        try:
            # If advanced settings exist in config, load them
            if 'advanced_cache_size' in self.config:
                self.cache_size_var.set(str(self.config['advanced_cache_size']))
            if 'advanced_cuda_split_size' in self.config:
                self.cuda_split_size_var.set(str(self.config['advanced_cuda_split_size']))
            if 'advanced_batch_size' in self.config:
                self.batch_size_var.set(str(self.config['advanced_batch_size']))
        except Exception as e:
            self.status_text.insert(tk.END, f"Failed to load advanced settings: {str(e)}\n")

    def apply_advanced_settings(self):
        """Apply advanced performance settings"""
        try:
            # Get advanced setting values
            cache_size = int(self.cache_size_var.get())
            cuda_split_size = int(self.cuda_split_size_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Save to configuration
            self.config['advanced_cache_size'] = cache_size
            self.config['advanced_cuda_split_size'] = cuda_split_size
            self.config['advanced_batch_size'] = batch_size
            
            # Update configuration file
            self.save_config()
            
            # Display confirmation message
            self.status_text.insert(tk.END, "\n===== Advanced Settings Applied =====\n")
            self.status_text.insert(tk.END, f"Memory Cache Size: {cache_size}\n")
            self.status_text.insert(tk.END, f"CUDA Memory Chunk Size: {cuda_split_size}MB\n")
            self.status_text.insert(tk.END, f"Batch Size: {batch_size}\n")
            self.status_text.insert(tk.END, "These settings will take effect on next server startup\n")
            self.status_text.see(tk.END)
            
            messagebox.showinfo("Success", "Advanced settings applied, will take effect on next server startup")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply advanced settings: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VLLMServerGUI(root)
    root.mainloop()

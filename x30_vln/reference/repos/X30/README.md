import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
import math
import socket # 确保导入
import struct # 确保导入

# --- RobotController 类 (与上一版本相同，此处省略以保持简洁) ---
class RobotController:
    """
    机器人控制器类，用于通过UDP与机器人进行通信和控制。
    """
    ROBOT_IP = "192.168.1.103"
    ROBOT_PORT = 43893
    HEARTBEAT_CMD = 0x21040001
    CONFIRM_HEARTBEAT_CMD = 0x21020001
    STAND_UP_DOWN_CMD = 0x21010202
    FORCE_CONTROL_MODE_CMD = 0x2101020A
    START_STOP_MOTION_CMD = 0x21010201
    SOFT_ESTOP_CMD = 0x21010C0E

    YAW_CMD = 0x21010135
    LEFT_STICK_Y_CMD = 0x21010130
    LEFT_STICK_X_CMD = 0x21010131
    BODY_HEIGHT_CMD = 0x21010406

    WALKING_GAIT_CMD = 0x21010300
    SLOPE_GAIT_CMD = 0x21010402
    OBSTACLE_GAIT_CMD = 0x21010401
    STAIRS_GAIT_CMD = 0x21010405 # 对应文档中的 "楼梯步态（实心地面/镂空地面/无踢面楼梯模式）"
    L_WALK_GAIT_CMD = 0x21010420
    MOUNTAIN_GAIT_CMD = 0x21010421
    SILENT_GAIT_CMD = 0x21010422

    AXIS_CMD_RATE_HZ = 50.0
    MAX_AXIS_VALUE = 30000
    MIN_AXIS_VALUE = -30000

    def __init__(self, robot_ip=ROBOT_IP, robot_port=ROBOT_PORT, gui_update_callback=None):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.sock = None
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()
        self.is_connected = False
        self.gui_update_callback = gui_update_callback
        self.movement_timers = {}
        self.current_velocities = {'vx': 0, 'vy': 0, 'wz': 0}
        self.current_speed_factor = 1.0

        self.gait_map_chinese_to_cmd = {
            "行走步态": self.WALKING_GAIT_CMD,
            "斜坡步态": self.SLOPE_GAIT_CMD,
            "越障步态": self.OBSTACLE_GAIT_CMD,
            "楼梯步态(标准)": self.STAIRS_GAIT_CMD,
            "L行走步态": self.L_WALK_GAIT_CMD,
            "山地步态": self.MOUNTAIN_GAIT_CMD,
            "静音步态": self.SILENT_GAIT_CMD,
        }
        self.ordered_gaits_chinese = [ # 保持一个有序列表以便按钮按此顺序创建
            "行走步态", "斜坡步态", "越障步态", "楼梯步态(标准)",
            "L行走步态", "山地步态", "静音步态"
        ]


    def _send_udp_command(self, cmd_code, cmd_value, cmd_type=0):
        if not self.sock:
            if self.gui_update_callback:
                self.gui_update_callback("log", "错误：套接字未初始化。")
            return False
        param_val_uint32 = cmd_value & 0xFFFFFFFF
        try:
            payload = struct.pack("<III", cmd_code, param_val_uint32, cmd_type)
            self.sock.sendto(payload, (self.robot_ip, self.robot_port))
            return True
        except socket.error as e:
            if self.gui_update_callback:
                self.gui_update_callback("log", f"发送命令 {hex(cmd_code)} 时发生套接字错误: {e}")
        except Exception as e:
            if self.gui_update_callback:
                 self.gui_update_callback("log", f"发送命令 {hex(cmd_code)} 时发生未知错误: {e}")
        return False

    def _heartbeat_thread_func(self):
        if self.gui_update_callback:
            self.gui_update_callback("log", "心跳线程已启动。")
        heartbeat_interval = 1.0 / 2.0 # 2Hz, 即0.5秒一次

        while not self.heartbeat_stop_event.is_set():
            if self.is_connected and self.sock:
                self._send_udp_command(self.HEARTBEAT_CMD, 0)
            self.heartbeat_stop_event.wait(heartbeat_interval)
        if self.gui_update_callback:
            self.gui_update_callback("log", "心跳线程已停止。")


    def connect(self):
        if self.is_connected:
            if self.gui_update_callback: self.gui_update_callback("log", "已经连接到机器人。")
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.gui_update_callback: self.gui_update_callback("log", f"尝试连接到机器人 {self.robot_ip}:{self.robot_port}...")
            self.heartbeat_stop_event.clear()
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_thread_func)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
            time.sleep(0.1)
            if not self._send_udp_command(self.CONFIRM_HEARTBEAT_CMD, 0):
                raise Exception("发送确认连接心跳失败")
            if self.gui_update_callback: self.gui_update_callback("log", "确认连接心跳(0x21020001)已发送。")
            self.is_connected = True
            if self.gui_update_callback: self.gui_update_callback("status", "已连接")
            if self.gui_update_callback: self.gui_update_callback("log", "成功连接到机器人。")
            return True
        except Exception as e:
            if self.gui_update_callback: self.gui_update_callback("log", f"连接机器人失败: {e}")
            if self.sock: self.sock.close(); self.sock = None
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_stop_event.set()
                self.heartbeat_thread.join(timeout=1)
            self.is_connected = False
            if self.gui_update_callback: self.gui_update_callback("status", "已断开")
            return False

    def disconnect(self):
        if not self.is_connected and not (self.heartbeat_thread and self.heartbeat_thread.is_alive()):
             if self.gui_update_callback: self.gui_update_callback("log", "已经断开连接。")
             return
        if self.gui_update_callback: self.gui_update_callback("log", "正在断开与机器人的连接...")
        self.stop_all_movement_continuous()
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_stop_event.set()
            self.heartbeat_thread.join(timeout=1.0)
            if self.heartbeat_thread.is_alive():
                if self.gui_update_callback: self.gui_update_callback("log", "警告: 心跳线程未能在超时内停止。")
            self.heartbeat_thread = None
        if self.sock:
            self.sock.close()
            self.sock = None
        self.is_connected = False
        if self.gui_update_callback: self.gui_update_callback("status", "已断开")
        if self.gui_update_callback: self.gui_update_callback("log", "已与机器人断开连接。")

    def stand_up(self):
        if self.gui_update_callback: self.gui_update_callback("log", "命令机器人站立...")
        self._send_udp_command(self.STAND_UP_DOWN_CMD, 0)
        if self.gui_update_callback: self.gui_update_callback("robot_state", "尝试站立")

    def lie_down(self):
        if self.gui_update_callback: self.gui_update_callback("log", "命令机器人趴下...")
        self._send_udp_command(self.STAND_UP_DOWN_CMD, 0)
        if self.gui_update_callback: self.gui_update_callback("robot_state", "尝试趴下")

    def enter_force_control_standing(self):
        if self.gui_update_callback: self.gui_update_callback("log", "命令进入力控站立状态...")
        self._send_udp_command(self.FORCE_CONTROL_MODE_CMD, 0)
        if self.gui_update_callback: self.gui_update_callback("robot_state", "尝试力控站立")

    def enter_stepping_mode(self):
        if self.gui_update_callback: self.gui_update_callback("log", "命令进入踏步状态...")
        self._send_udp_command(self.START_STOP_MOTION_CMD, 0)

    def exit_stepping_mode(self):
        if self.gui_update_callback: self.gui_update_callback("log", "命令退出踏步状态...")
        self._send_udp_command(self.START_STOP_MOTION_CMD, 0)

    def set_speed_factor(self, factor):
        if 0.0 <= factor <= 1.0:
            self.current_speed_factor = factor
            if self.gui_update_callback:
                self.gui_update_callback("log", f"移动速度已设置为: {int(factor*100)}%")
        else:
            if self.gui_update_callback:
                self.gui_update_callback("log", f"错误：无效的速度因子 {factor}，应在0.0到1.0之间。")

    def start_movement_continuous(self, direction):
        if not self.is_connected: return
        if self.gui_update_callback: self.gui_update_callback("log", f"开始向 {direction} 持续移动 (速度: {int(self.current_speed_factor*100)}%)")
        if direction in self.movement_timers:
            timer = self.movement_timers.pop(direction)
            if timer: timer.cancel()
        scaled_positive_value = int(self.current_speed_factor * self.MAX_AXIS_VALUE)
        scaled_negative_value = int(self.current_speed_factor * self.MIN_AXIS_VALUE)
        value_to_set_in_velocity = 0
        cmd_to_use = None
        if direction == "forward": value_to_set_in_velocity = scaled_positive_value; cmd_to_use = self.LEFT_STICK_Y_CMD; self.current_velocities['vx'] = value_to_set_in_velocity
        elif direction == "backward": value_to_set_in_velocity = scaled_negative_value; cmd_to_use = self.LEFT_STICK_Y_CMD; self.current_velocities['vx'] = value_to_set_in_velocity
        elif direction == "left_strafe": value_to_set_in_velocity = scaled_negative_value; cmd_to_use = self.LEFT_STICK_X_CMD; self.current_velocities['vy'] = value_to_set_in_velocity
        elif direction == "right_strafe": value_to_set_in_velocity = scaled_positive_value; cmd_to_use = self.LEFT_STICK_X_CMD; self.current_velocities['vy'] = value_to_set_in_velocity
        elif direction == "turn_left": value_to_set_in_velocity = scaled_negative_value; cmd_to_use = self.YAW_CMD; self.current_velocities['wz'] = value_to_set_in_velocity
        elif direction == "turn_right": value_to_set_in_velocity = scaled_positive_value; cmd_to_use = self.YAW_CMD; self.current_velocities['wz'] = value_to_set_in_velocity
        else: return
        def send_and_reschedule():
            if direction not in self.movement_timers or not self.is_connected: return
            current_cmd_value_to_send = 0
            if cmd_to_use == self.LEFT_STICK_Y_CMD: current_cmd_value_to_send = self.current_velocities['vx']
            elif cmd_to_use == self.LEFT_STICK_X_CMD: current_cmd_value_to_send = self.current_velocities['vy']
            elif cmd_to_use == self.YAW_CMD: current_cmd_value_to_send = self.current_velocities['wz']
            if current_cmd_value_to_send != 0 and self.current_speed_factor > 0.001 :
                 self._send_udp_command(cmd_to_use, current_cmd_value_to_send)
            if direction in self.movement_timers:
                 self.movement_timers[direction] = threading.Timer(1.0 / self.AXIS_CMD_RATE_HZ, send_and_reschedule)
                 self.movement_timers[direction].daemon = True
                 self.movement_timers[direction].start()
        if value_to_set_in_velocity != 0 and self.current_speed_factor > 0.001:
            self._send_udp_command(cmd_to_use, value_to_set_in_velocity)
        if self.current_speed_factor > 0.001:
            self.movement_timers[direction] = threading.Timer(1.0 / self.AXIS_CMD_RATE_HZ, send_and_reschedule)
            self.movement_timers[direction].daemon = True
            self.movement_timers[direction].start()
        elif direction in self.movement_timers :
            timer = self.movement_timers.pop(direction)
            if timer: timer.cancel()
            self._send_udp_command(cmd_to_use, 0)

    def stop_movement_continuous(self, direction):
        if self.gui_update_callback: self.gui_update_callback("log", f"停止向 {direction} 持续移动")
        if direction in self.movement_timers:
            timer = self.movement_timers.pop(direction)
            if timer: timer.cancel()
        cmd_to_send_stop = None
        if direction == "forward" or direction == "backward":
            self.current_velocities['vx'] = 0
            cmd_to_send_stop = self.LEFT_STICK_Y_CMD
        elif direction == "left_strafe" or direction == "right_strafe":
            self.current_velocities['vy'] = 0
            cmd_to_send_stop = self.LEFT_STICK_X_CMD
        elif direction == "turn_left" or direction == "turn_right":
            self.current_velocities['wz'] = 0
            cmd_to_send_stop = self.YAW_CMD
        if cmd_to_send_stop and self.is_connected:
            self._send_udp_command(cmd_to_send_stop, 0)

    def stop_all_movement_continuous(self):
        if self.gui_update_callback: self.gui_update_callback("log", "停止所有持续移动")
        for direction in list(self.movement_timers.keys()):
            self.stop_movement_continuous(direction)
        if self.is_connected:
            self._send_udp_command(self.LEFT_STICK_Y_CMD, 0)
            self._send_udp_command(self.LEFT_STICK_X_CMD, 0)
            self._send_udp_command(self.YAW_CMD, 0)
        self.current_velocities = {'vx': 0, 'vy': 0, 'wz': 0}

    def set_body_height(self, height_mode_str):
        value = 0
        if height_mode_str == "normal": value = 2
        elif height_mode_str == "crawl": value = 0
        else:
            if self.gui_update_callback: self.gui_update_callback("log", f"未知的高度模式: {height_mode_str}")
            return
        if self.gui_update_callback: self.gui_update_callback("log", f"设置身体高度为: {height_mode_str} (指令值: {value})")
        self._send_udp_command(self.BODY_HEIGHT_CMD, value)

    def set_gait(self, gait_name_chinese):
        if gait_name_chinese in self.gait_map_chinese_to_cmd:
            cmd_code = self.gait_map_chinese_to_cmd[gait_name_chinese]
            self._send_udp_command(cmd_code, 0)
            if self.gui_update_callback:
                self.gui_update_callback("log", f"尝试切换到步态: {gait_name_chinese}")
                # 实际的步态状态由监听器更新，这里可以触发一个预期的更新
                self.gui_update_callback("gait_selection_changed", gait_name_chinese)
        else:
            if self.gui_update_callback:
                self.gui_update_callback("log", f"未知的步态名称: {gait_name_chinese}")

    def trigger_soft_estop(self):
        if not self.is_connected:
            if self.gui_update_callback:
                self.gui_update_callback("log", "错误：未连接到机器人，无法发送软急停。")
            return False
        if self.gui_update_callback:
            self.gui_update_callback("log", "!!! 发送软急停指令 (0x21010C0E) !!!")
        self.stop_all_movement_continuous()
        sent = self._send_udp_command(self.SOFT_ESTOP_CMD, 0, 0)
        if sent and self.gui_update_callback:
            self.gui_update_callback("log", "软急停指令已发送。机器人应进入急停状态。")
        elif not sent and self.gui_update_callback:
            self.gui_update_callback("log", "软急停指令发送失败。")
        return sent

# --- 机器人状态监听器 (与上一版本相同，此处省略以保持简洁) ---
class RobotStateListener(threading.Thread):
    def __init__(self, gui_update_callback, listen_ip="", listen_port=43897): # 默认监听端口
        super().__init__()
        self.daemon = True
        self.gui_update_callback = gui_update_callback
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.sock = None
        self.running = False

    def run(self):
        self.running = True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.listen_ip, self.listen_port))
            self.sock.settimeout(1.0)
            if self.gui_update_callback: self.gui_update_callback("log", f"状态监听器已启动，监听地址: {self.listen_ip if self.listen_ip else '所有接口'}:{self.listen_port}")

            while self.running:
                try:
                    if not self.running: break
                    data, addr = self.sock.recvfrom(2048)
                    if not self.running: break
                    self.parse_and_update_state(data)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running and self.gui_update_callback:
                        self.gui_update_callback("log", f"状态监听器接收/解析错误: {e}")
        except OSError as e:
             if self.gui_update_callback: self.gui_update_callback("log", f"状态监听器绑定失败 (OSError): {e}. 请检查端口 {self.listen_port} 是否被占用或配置正确。")
        except Exception as e:
            if self.gui_update_callback: self.gui_update_callback("log", f"状态监听器启动时发生严重错误: {e}")
        finally:
            if self.sock: self.sock.close()
            if self.gui_update_callback: self.gui_update_callback("log", "状态监听器已停止。")

    def stop(self):
        self.running = False

    def parse_and_update_state(self, raw_data):
        if len(raw_data) < 12:
            return
        header_format = "<III"
        try:
            cmd_code, params_size, cmd_type = struct.unpack(header_format, raw_data[:12])
        except struct.error as e:
            if self.gui_update_callback:
                self.gui_update_callback("log", f"状态监听器: 解析头部失败: {e}, 数据: {raw_data[:12].hex()}")
            return
        payload = raw_data[12:]
        if len(payload) < params_size:
            return
        payload = payload[:params_size]

        if cmd_type == 1:
            if cmd_code == 0x1008:
                rcs_data_format = "<15siiiiii4f10BI"
                expected_len = struct.calcsize(rcs_data_format)
                if params_size >= expected_len:
                    try:
                        data = struct.unpack(rcs_data_format, payload[:expected_len])
                        robot_name_bytes = data[0]
                        current_mileage_cm = data[1]; total_mileage_cm = data[2]
                        rcs_state_bytes = data[11:21]; error_state_val = data[21]
                        robot_name = robot_name_bytes.split(b'\0', 1)[0].decode('utf-8', errors='replace')
                        is_nav_mode = rcs_state_bytes[0]
                        control_mode_str = "非手动(导航/辅助)" if is_nav_mode == 1 else "手动"
                        errors = []
                        if error_state_val & 0x01: errors.append("IMU超时")
                        if error_state_val & 0x02: errors.append("心跳超时")
                        if error_state_val & 0x04: errors.append("驱动器过温")
                        if error_state_val & 0x08: errors.append("驱动器故障")
                        if error_state_val & 0x10: errors.append("电机过温")
                        if error_state_val & 0x20: errors.append("电池低电量")
                        error_str = ", ".join(errors) if errors else "无"
                        if self.gui_update_callback:
                            self.gui_update_callback("robot_name", robot_name)
                            self.gui_update_callback("control_mode", control_mode_str)
                            self.gui_update_callback("error_status", error_str)
                            self.gui_update_callback("mileage_info", f"{current_mileage_cm/100.0:.2f}m / {total_mileage_cm/100.0:.2f}m")
                    except Exception as e:
                        if self.gui_update_callback: self.gui_update_callback("log", f"解析0x1008失败: {e}")
            elif cmd_code == 0x1009:
                motion_state_format = "<BBff3f3ffII10B"
                expected_len = struct.calcsize(motion_state_format)
                if params_size >= expected_len:
                    try:
                        s_data = struct.unpack(motion_state_format, payload[:expected_len])
                        basic_state_val, gait_state_val = s_data[0], s_data[1]
                        basic_state_map = {0: "趴下", 1: "正在起立", 2: "初始站立", 3: "力控站立", 4: "踏步", 5: "正在趴下", 6: "软急停/摔倒", 16: "RL状态"}
                        # 确保这里的中文名与 RobotController 中的 gait_map_chinese_to_cmd 的键一致
                        gait_map_feedback = {
                            0: "行走步态", 1: "越障步态", 2: "斜坡步态", 3: "跑步步态", # 跑步步态可能不在按钮中
                            6: "楼梯步态(标准)", 7: "楼梯(累积帧)", 8: "45°楼梯(累积帧)", # 这些可能需要映射到按钮上的名称
                            32: "L行走步态", 33: "山地步态", 34: "静音步态"
                        }
                        current_basic_state = basic_state_map.get(basic_state_val, f"基础({basic_state_val})")
                        current_gait_from_robot = gait_map_feedback.get(gait_state_val, f"步态值({gait_state_val})")

                        if self.gui_update_callback:
                            self.gui_update_callback("robot_state", current_basic_state)
                            self.gui_update_callback("gait_state", current_gait_from_robot) # 发送机器人实际回报的步态
                    except Exception as e:
                        if self.gui_update_callback: self.gui_update_callback("log", f"解析0x1009失败: {e}")
            elif cmd_code == 0x100B:
                controller_safe_format = "<12f12Bff"
                expected_len = struct.calcsize(controller_safe_format)
                if params_size >= expected_len:
                    try:
                        s_data = struct.unpack(controller_safe_format, payload[:expected_len])
                        motor_temps = s_data[0:12]
                        avg_motor_temp = sum(motor_temps) / len(motor_temps) if motor_temps else 0.0
                        if self.gui_update_callback:
                            self.gui_update_callback("motor_temp_avg", f"{avg_motor_temp:.1f}°C")
                    except Exception as e:
                        if self.gui_update_callback: self.gui_update_callback("log", f"解析0x100B失败: {e}")
            elif cmd_code == 0x21050F0A:
                battery_level_offset = 19
                if params_size >= battery_level_offset + 1:
                    try:
                        battery_percent_val = struct.unpack("<B", payload[battery_level_offset:battery_level_offset+1])[0]
                        if self.gui_update_callback:
                            self.gui_update_callback("battery_level", f"电量: {battery_percent_val}%")
                    except Exception as e:
                         if self.gui_update_callback: self.gui_update_callback("log", f"解析0x21050F0A电量失败: {e}")
        elif cmd_type == 0:
            if cmd_code == 0x11050F08:
                 height_val_from_param_size = params_size
                 if height_val_from_param_size == 0xFFFFFFFF: height_val_signed = -1
                 else: height_val_signed = height_val_from_param_size
                 current_height_state_simple = "高度未知"
                 if height_val_signed == 0: current_height_state_simple = "身体高度：正常"
                 elif height_val_signed == -1: current_height_state_simple = "身体高度：匍匐"
                 else: current_height_state_simple = f"高度({hex(height_val_from_param_size)})"
                 if self.gui_update_callback:
                    self.gui_update_callback("body_height_state", current_height_state_simple)

# --- GUI 应用 ---
class RobotControlGUI:
    def __init__(self, master):
        self.master = master
        master.title("机器人控制系统")
        master.geometry("1150x850") # 可能需要调整高度

        self.controller = RobotController(gui_update_callback=self.update_gui_from_main_thread)
        self.state_listener = None
        self.gait_buttons = {} # 用于存储步态按钮，方便管理状态
        self.current_selected_gait_button = None # 当前选中的步态按钮
        self.default_gait = "行走步态" # 定义默认步态

        style = ttk.Style(master)
        style.configure("Emergency.TButton", foreground="white", background="red", font=('Helvetica', 10, 'bold'), relief="raised", padding=6)
        style.map("Emergency.TButton", background=[('active', 'darkred')])
        # 选中步态按钮的样式 (ttk按钮的背景色可能不直接生效，relief是更可靠的方式)
        style.configure("SelectedGait.TButton", relief="sunken", foreground="blue")
        style.configure("NormalGait.TButton", relief="raised")


        connection_frame = ttk.LabelFrame(master, text="连接与状态")
        connection_frame.pack(padx=10, pady=10, fill="x")
        self.connect_button = ttk.Button(connection_frame, text="连接机器人", command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.estop_button = ttk.Button(connection_frame, text="软急停 (Backspace)", style="Emergency.TButton", command=self.trigger_emergency_stop)
        self.estop_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.connection_status_label = ttk.Label(connection_frame, text="状态: 已断开", width=12)
        self.connection_status_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.robot_state_label = ttk.Label(connection_frame, text="机器人状态: 未知", width=15)
        self.robot_state_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.gait_state_label = ttk.Label(connection_frame, text="当前步态: 未知", width=18) # 这个标签仍然显示机器人实际的步态
        self.gait_state_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.body_height_label = ttk.Label(connection_frame, text="身体高度: 未知", width=15)
        self.body_height_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.battery_level_label = ttk.Label(connection_frame, text="电量: 未知", width=12)
        self.battery_level_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.control_mode_label = ttk.Label(connection_frame, text="控制模式: 未知", width=18)
        self.control_mode_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.error_status_label = ttk.Label(connection_frame, text="错误: 无", width=20, wraplength=150)
        self.error_status_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.motor_temp_label = ttk.Label(connection_frame, text="电机均温: 未知", width=15)
        self.motor_temp_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.mileage_label = ttk.Label(connection_frame, text="里程: 未知", width=20)
        self.mileage_label.pack(side=tk.LEFT, padx=5, pady=5)

        row1_frame = ttk.Frame(master)
        row1_frame.pack(fill="x", padx=10, pady=5)
        pose_frame = ttk.LabelFrame(row1_frame, text="姿态控制")
        pose_frame.pack(side=tk.LEFT, padx=0, pady=5, fill="x", expand=True)
        ttk.Button(pose_frame, text="站立", command=self.controller.stand_up).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(pose_frame, text="趴下", command=self.controller.lie_down).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(pose_frame, text="进入力控", command=self.controller.enter_force_control_standing).pack(side=tk.LEFT, padx=5, pady=2)
        mode_frame = ttk.LabelFrame(row1_frame, text="运动模式与高度")
        mode_frame.pack(side=tk.LEFT, padx=10, pady=5, fill="x", expand=True)
        ttk.Button(mode_frame, text="进入踏步模式", command=self.controller.enter_stepping_mode).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(mode_frame, text="退出踏步模式", command=self.controller.exit_stepping_mode).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(mode_frame, text="高度:").pack(side=tk.LEFT, padx=(10,0), pady=2)
        self.height_var = tk.StringVar(value="normal")
        ttk.Radiobutton(mode_frame, text="正常", variable=self.height_var, value="normal", command=self.set_height).pack(side=tk.LEFT, pady=2)
        ttk.Radiobutton(mode_frame, text="匍匐", variable=self.height_var, value="crawl", command=self.set_height).pack(side=tk.LEFT, pady=2)

        row2_frame = ttk.Frame(master)
        row2_frame.pack(fill="x", padx=10, pady=5)
        movement_frame = ttk.LabelFrame(row2_frame, text="移动控制 (踏步模式下)")
        movement_frame.pack(side=tk.LEFT, padx=0, pady=5, fill="x", expand=True)
        btn_fwd = ttk.Button(movement_frame, text="前进 (W)")
        btn_fwd.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        btn_fwd.bind("<ButtonPress>", lambda e: self.controller.start_movement_continuous("forward"))
        btn_fwd.bind("<ButtonRelease>", lambda e: self.controller.stop_movement_continuous("forward"))
        btn_left_turn = ttk.Button(movement_frame, text="左转 (A)")
        btn_left_turn.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        btn_left_turn.bind("<ButtonPress>", lambda e: self.controller.start_movement_continuous("turn_left"))
        btn_left_turn.bind("<ButtonRelease>", lambda e: self.controller.stop_movement_continuous("turn_left"))
        btn_bwd = ttk.Button(movement_frame, text="后退 (S)")
        btn_bwd.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        btn_bwd.bind("<ButtonPress>", lambda e: self.controller.start_movement_continuous("backward"))
        btn_bwd.bind("<ButtonRelease>", lambda e: self.controller.stop_movement_continuous("backward"))
        btn_right_turn = ttk.Button(movement_frame, text="右转 (D)")
        btn_right_turn.grid(row=1, column=2, padx=5, pady=2, sticky="ew")
        btn_right_turn.bind("<ButtonPress>", lambda e: self.controller.start_movement_continuous("turn_right"))
        btn_right_turn.bind("<ButtonRelease>", lambda e: self.controller.stop_movement_continuous("turn_right"))
        btn_left_strafe = ttk.Button(movement_frame, text="左移 (Q)")
        btn_left_strafe.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        btn_left_strafe.bind("<ButtonPress>", lambda e: self.controller.start_movement_continuous("left_strafe"))
        btn_left_strafe.bind("<ButtonRelease>", lambda e: self.controller.stop_movement_continuous("left_strafe"))
        btn_right_strafe = ttk.Button(movement_frame, text="右移 (E)")
        btn_right_strafe.grid(row=0, column=2, padx=5, pady=2, sticky="ew")
        btn_right_strafe.bind("<ButtonPress>", lambda e: self.controller.start_movement_continuous("right_strafe"))
        btn_right_strafe.bind("<ButtonRelease>", lambda e: self.controller.stop_movement_continuous("right_strafe"))
        movement_frame.columnconfigure((0,1,2), weight=1)
        speed_frame = ttk.LabelFrame(row2_frame, text="速度设置")
        speed_frame.pack(side=tk.LEFT, padx=10, pady=5, fill="x", expand=True)
        ttk.Label(speed_frame, text="速度 (%):").pack(side=tk.LEFT, padx=5, pady=2)
        self.speed_scale_var = tk.DoubleVar(value=50.0)
        self.controller.set_speed_factor(0.5)
        self.speed_scale = ttk.Scale(speed_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.speed_scale_var, command=self.update_speed_factor_from_scale)
        self.speed_scale.pack(side=tk.LEFT, padx=5, pady=2, fill="x", expand=True)
        self.speed_label = ttk.Label(speed_frame, text="50%")
        self.speed_label.pack(side=tk.LEFT, padx=5, pady=2)

        # --- 步态选择按钮 ---
        gait_button_frame = ttk.LabelFrame(master, text="步态选择 (踏步模式下)")
        gait_button_frame.pack(padx=10, pady=5, fill="x")

        # 将按钮放在多行，如果一行放不下
        max_buttons_per_row = 4
        current_row_frame = None

        for i, gait_name_chinese in enumerate(self.controller.ordered_gaits_chinese):
            if i % max_buttons_per_row == 0:
                current_row_frame = ttk.Frame(gait_button_frame)
                current_row_frame.pack(fill="x")

            button = ttk.Button(current_row_frame, text=gait_name_chinese,
                                command=lambda gn=gait_name_chinese: self.set_gait_from_button(gn),
                                style="NormalGait.TButton")
            button.pack(side=tk.LEFT, padx=5, pady=2, fill="x", expand=True)
            self.gait_buttons[gait_name_chinese] = button

        # 设置默认选中的步态
        self.set_selected_gait_button(self.default_gait)


        log_frame = ttk.LabelFrame(master, text="日志")
        log_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill="both", expand=True)

        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.key_pressed_states = {}
        self.bind_keyboard_shortcuts()

    def set_selected_gait_button(self, gait_name_to_select):
        """ 更新步态按钮的视觉状态 """
        for gait_name, btn in self.gait_buttons.items():
            if gait_name == gait_name_to_select:
                btn.config(style="SelectedGait.TButton")
                self.current_selected_gait_button = btn
            else:
                btn.config(style="NormalGait.TButton")

    def set_gait_from_button(self, selected_gait_chinese):
        if not self.controller.is_connected:
            messagebox.showwarning("提示", "请先连接机器人")
            return
        self.controller.set_gait(selected_gait_chinese)
        # 乐观更新按钮状态，实际状态由机器人反馈更新gait_state_label
        self.set_selected_gait_button(selected_gait_chinese)


    def trigger_emergency_stop(self):
        if not self.controller.is_connected:
            messagebox.showwarning("提示", "机器人未连接，无法发送软急停。")
            return
        if messagebox.askokcancel("软急停确认", "您确定要执行软急停吗？这将停止机器人所有动作。"):
            self.controller.trigger_soft_estop()

    def bind_keyboard_shortcuts(self):
        key_map = {
            'w': "forward", 's': "backward",
            'a': "turn_left", 'd': "turn_right",
            'q': "left_strafe", 'e': "right_strafe",
        }
        for key_char, direction in key_map.items():
            self.master.bind(f"<{key_char.lower()}>", lambda e, d=direction: self.handle_key_press(d))
            self.master.bind(f"<KeyRelease-{key_char.lower()}>", lambda e, d=direction: self.handle_key_release(d))
            self.master.bind(f"<{key_char.upper()}>", lambda e, d=direction: self.handle_key_press(d))
            self.master.bind(f"<KeyRelease-{key_char.upper()}>", lambda e, d=direction: self.handle_key_release(d))
        self.key_pressed_states = {direction: False for direction in key_map.values()}
        self.master.bind("<BackSpace>", lambda event: self.trigger_emergency_stop_from_key())

    def trigger_emergency_stop_from_key(self):
        if not self.controller.is_connected:
            self.update_gui_from_main_thread("log", "键盘急停：机器人未连接。")
            return
        self.update_gui_from_main_thread("log", "通过键盘 Backspace 触发软急停...")
        self.controller.trigger_soft_estop()

    def handle_key_press(self, direction):
        if self.controller.is_connected and not self.key_pressed_states.get(direction, False):
            self.controller.start_movement_continuous(direction)
            self.key_pressed_states[direction] = True

    def handle_key_release(self, direction):
        if self.controller.is_connected:
            self.controller.stop_movement_continuous(direction)
            self.key_pressed_states[direction] = False

    def toggle_connection(self):
        if not self.controller.is_connected:
            if self.controller.connect():
                self.connect_button.config(text="断开连接")
                if not self.state_listener or not self.state_listener.is_alive():
                    self.state_listener = RobotStateListener(self.update_gui_from_main_thread, listen_port=43897)
                    self.state_listener.start()
                # 连接成功后，尝试设置默认步态（如果机器人支持立即切换）
                # 或者等待机器人状态反馈后，根据反馈的步态更新按钮选中状态
                self.set_selected_gait_button(self.default_gait) # 初始选中
            else:
                 messagebox.showerror("连接失败", "无法连接到机器人。检查IP、端口和网络。")
        else:
            if self.state_listener and self.state_listener.is_alive():
                self.state_listener.stop()
                self.state_listener = None
            self.controller.disconnect()
            self.connect_button.config(text="连接机器人")
            self.update_gui_from_main_thread("robot_state", "未知")
            self.update_gui_from_main_thread("gait_state", "未知") # 重置标签
            self.set_selected_gait_button(self.default_gait) # 断开连接也重置按钮到默认
            self.update_gui_from_main_thread("body_height_state", "身体高度: 未知")
            self.update_gui_from_main_thread("battery_level", "电量: 未知")
            self.update_gui_from_main_thread("control_mode", "控制模式: 未知")
            self.update_gui_from_main_thread("error_status", "错误: 无")
            self.update_gui_from_main_thread("motor_temp_avg", "电机均温: 未知")
            self.update_gui_from_main_thread("mileage_info", "里程: 未知")
            self.master.title("机器人控制系统")

    def set_height(self):
        if not self.controller.is_connected: messagebox.showwarning("提示", "请先连接机器人"); return
        selected_height = self.height_var.get()
        self.controller.set_body_height(selected_height)

    # 移除 set_gait_from_menu 方法，因为它不再被使用

    def update_speed_factor_from_scale(self, scale_value_str):
        try:
            scale_value = float(scale_value_str)
            factor = scale_value / 100.0
            self.controller.set_speed_factor(factor)
            self.speed_label.config(text=f"{int(scale_value)}%")
        except ValueError:
            if self.controller.gui_update_callback:
                self.controller.gui_update_callback("log", f"错误: 无效的速度标度值 {scale_value_str}")

    def update_gui_from_main_thread(self, type, message):
        if self.master and self.master.winfo_exists():
            self.master.after(0, self._do_update_gui, type, message)

    def _do_update_gui(self, type, message):
        if not self.master or not self.master.winfo_exists(): return

        if type == "log":
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        elif type == "status": self.connection_status_label.config(text=f"状态: {message}")
        elif type == "robot_state": self.robot_state_label.config(text=f"机器人状态: {message}")
        elif type == "gait_state": # 这个消息来自机器人状态监听器
            self.gait_state_label.config(text=f"当前步态: {message}")
            # 根据机器人实际回报的步态，更新按钮的选中状态
            # 需要确保 message (机器人回报的步态名) 与按钮上的文本匹配
            # 如果 RobotStateListener 中 gait_map_feedback 的值与按钮文本不完全一致，需要适配
            if message in self.gait_buttons:
                self.set_selected_gait_button(message)
            else:
                # 如果机器人回报的步态名不在我们的按钮列表中，取消所有按钮的特殊选中状态
                # 或者标记为一个“未知”或“其他”状态（如果需要）
                if self.current_selected_gait_button:
                    self.current_selected_gait_button.config(style="NormalGait.TButton")
                    self.current_selected_gait_button = None
                # self.gait_state_label.config(text=f"当前步态: {message} (非预设)")
        elif type == "gait_selection_changed": # 这个消息来自controller，是用户点击按钮的预期
            # 当用户点击按钮时，我们已经乐观地更新了按钮的视觉状态
            # 这里的消息主要是为了记录日志，实际的标签和按钮状态应由 "gait_state" (机器人反馈) 驱动
            pass # self.update_gui_from_main_thread("log", f"用户选择步态: {message}")
        elif type == "body_height_state": self.body_height_label.config(text=message)
        elif type == "battery_level": self.battery_level_label.config(text=message)
        elif type == "control_mode": self.control_mode_label.config(text=f"控制模式: {message}")
        elif type == "error_status": self.error_status_label.config(text=f"错误: {message}")
        elif type == "motor_temp_avg": self.motor_temp_label.config(text=f"电机均温: {message}")
        elif type == "mileage_info": self.mileage_label.config(text=f"里程: {message}")
        elif type == "robot_name" and message:
            self.master.title(f"机器人控制系统 - {message}")

    def on_closing(self):
        if messagebox.askokcancel("退出", "确定要退出机器人控制系统吗?"):
            if self.state_listener and self.state_listener.is_alive():
                self.state_listener.stop()
                self.state_listener.join(timeout=1.0)
                if self.state_listener.is_alive():
                     self.update_gui_from_main_thread("log","警告: 状态监听线程未能正常停止。")
                self.state_listener = None
            if self.controller.is_connected:
                self.controller.disconnect()
            if self.controller.heartbeat_thread and self.controller.heartbeat_thread.is_alive():
                 self.controller.heartbeat_stop_event.set()
                 self.controller.heartbeat_thread.join(timeout=0.5)
            self.master.destroy()

def main_gui():
    root = tk.Tk()
    app = RobotControlGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main_gui()# X30
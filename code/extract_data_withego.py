import json
import math
import numpy as np
from nuscenes.nuscenes import NuScenes

# --- Configuration ---
questions_file = 'NuScenes_train_questions.json'
dataroot = 'nuscenes'
version = 'v1.0-trainval'
ego_channel = 'LIDAR_TOP'   # 用这个传感器流对齐 ego_pose

# ---------- 辅助：基于 ego_channel 的相邻 ego_pose 做差分，估计 v_ego ----------
def _yaw_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    """四元数(w,x,y,z) -> yaw(绕z轴, 弧度)，ZYX欧拉"""
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def ego_velocity_from_sample(nusc, sample_token: str, channel: str = 'LIDAR_TOP') :
    """
    返回 (v_global, v_dir2d):
      - v_global: [vx, vy, vz] m/s（全局系），用中心差分/单边差分估计（与原先一致）
      - v_dir2d: 归一化“前进方向” [fx, fy]（全局系），由当前帧 ego_pose 的 yaw 计算（与速度大小无关）
    """
    try:
        sample = nusc.get('sample', sample_token)
        sd_cur = nusc.get('sample_data', sample['data'][channel])

        def get_pose(sd_token):
            sd = nusc.get('sample_data', sd_token)
            pose = nusc.get('ego_pose', sd['ego_pose_token'])
            p = np.array(pose['translation'], float)           # [tx, ty, tz]
            t = float(pose['timestamp']) * 1e-6                # μs -> s
            return p, t

        # ---- 1) 速度 v_global：保持你的差分逻辑不变 ----
        if sd_cur['prev'] and sd_cur['next']:
            p0, t0 = get_pose(sd_cur['prev'])
            p1, t1 = get_pose(sd_cur['next'])
        else:
            other = sd_cur['next'] or sd_cur['prev']
            if not other:
                v_global = None
            else:
                p0, t0 = get_pose(sd_cur['token'])
                p1, t1 = get_pose(other)
            if other:
                dt = float(t1 - t0)
                v_global = ((p1 - p0) / dt).astype(float).tolist() if dt > 0 else None
        # 若两侧都有，则上面已算；否则 v_global 可能为 None
        if sd_cur['prev'] and sd_cur['next']:
            dt = float(t1 - t0)
            v_global = ((p1 - p0) / dt).astype(float).tolist() if dt > 0 else None

        # ---- 2) 方向 v_dir2d：用当前帧 ego_pose 的 yaw ----
        pose_cur = nusc.get('ego_pose', sd_cur['ego_pose_token'])
        qw, qx, qy, qz = map(float, pose_cur['rotation'])      # (w, x, y, z)
        yaw = _yaw_from_quat_wxyz(qw, qx, qy, qz)
        v_dir2d = [math.cos(yaw), math.sin(yaw)]               # 全局系：x前 / y左 / z上 的前向投影

        return v_global, v_dir2d

    except Exception:
        return None, None
# --- Load Questions ---
try:
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)['questions']
    print(f"Successfully loaded questions from {questions_file}")
except FileNotFoundError:
    print(f"Error: Questions file not found at {questions_file}")
    questions_data = []
print(f"Total questions loaded: {len(questions_data)}")

# --- Initialize NuScenes ---
try:
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    print("NuScenes dataset initialized successfully.")
except Exception as e:
    print(f"Error initializing NuScenes dataset: {e}")
    nusc = None

# --- Process Questions and Extract Annotations ---
if nusc and questions_data:
    # Extract all sample tokens from questions
    question_sample_tokens = set(q.get('sample_token') for q in questions_data if q.get('sample_token'))
    print(f"Unique sample tokens in questions: {len(question_sample_tokens)}")

    # Available sample tokens
    available_sample_tokens = set(s['token'] for s in nusc.sample)
    print(f"Available sample tokens in nuScenes dataset: {len(available_sample_tokens)}")

    # Missing vs found
    missing_tokens = question_sample_tokens - available_sample_tokens
    found_tokens = question_sample_tokens & available_sample_tokens
    print("\n--- Token Analysis ---")
    print(f"Tokens found in dataset: {len(found_tokens)}")
    print(f"Tokens missing from dataset: {len(missing_tokens)}")

    if missing_tokens:
        missing_tokens_file = 'missing_tokens.json'
        with open(missing_tokens_file, 'w') as f:
            json.dump(list(missing_tokens), f, indent=4)
        print(f"\nMissing tokens saved to {missing_tokens_file}")

# --- Process Questions and Extract Annotations (with ego_pose) ---
extracted_data = {}

if nusc and questions_data:
    processed_sample_tokens = set()
    successful_extractions = 0
    failed_extractions = 0

    for question_info in questions_data:
        sample_token = question_info.get('sample_token')
        if not sample_token or sample_token in processed_sample_tokens:
            continue

        processed_sample_tokens.add(sample_token)
        try:
            sample_record = nusc.get('sample', sample_token)
            annotation_tokens = sample_record.get('anns', [])

            # 取与 sample 对齐的 ego_pose（基于 LIDAR_TOP）
            try:
                sd_token = sample_record['data'][ego_channel]
                sd_rec = nusc.get('sample_data', sd_token)
                ego_pose = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                ego_pose_translation = ego_pose.get('translation', None)
                ego_pose_rotation = ego_pose.get('rotation', None)
                ego_pose_timestamp = ego_pose.get('timestamp', None)
                ego_pose_timestamp_sec = ego_pose_timestamp * 1e-6 if ego_pose_timestamp is not None else None
            except Exception as e:
                ego_pose_translation = None
                ego_pose_rotation = None
                ego_pose_timestamp_sec = None
                print(f"Warning: failed to fetch ego_pose for sample {sample_token}: {e}")

            # （可选）估计 v_ego
            v_ego_global, v_ego_dir2d = ego_velocity_from_sample(nusc, sample_token, channel=ego_channel)

            sample_annotations_data = []
            for ann_token in annotation_tokens:
                try:
                    ann = nusc.get('sample_annotation', ann_token)
                    annotation_details = {
                        'annotation_token': ann_token,
                        'category_name': ann.get('category_name'),
                        'translation': ann.get('translation'),
                        'size': ann.get('size'),
                        'rotation': ann.get('rotation'),
                        'num_lidar_pts': ann.get('num_lidar_pts'),
                        'num_radar_pts': ann.get('num_radar_pts'),
                        'visibility_level': nusc.get('visibility', ann.get('visibility_token', {})).get('level') if ann.get('visibility_token') else None,
                        'attribute_names': [nusc.get('attribute', at).get('name') for at in ann.get('attribute_tokens', [])],

                        # >>> 新增：把 ego 信息直接写进每条 annotation <<<
                        'ego_pose_translation': ego_pose_translation,
                        'ego_pose_rotation': ego_pose_rotation,
                        'ego_pose_timestamp': ego_pose_timestamp_sec,
                        'ego_v_global': v_ego_global,   # [vx,vy,vz] or None
                        'ego_v_dir2d': v_ego_dir2d,     # [fx,fy] or None
                    }
                    sample_annotations_data.append(annotation_details)
                except Exception as e:
                    print(f"Error processing annotation {ann_token} for sample {sample_token}: {e}")

            extracted_data[sample_token] = sample_annotations_data
            successful_extractions += 1

        except Exception as e:
            print(f"Error processing sample token {sample_token}: {e}")
            failed_extractions += 1

    print(f"\n--- Processing Summary ---")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")

# --- Output Results ---
print("\n--- Extracted Annotation Data ---")
print(f"Total samples processed: {len(extracted_data)}")
output_file = 'data/extracted_train_annotations_wego.json'
with open(output_file, 'w') as outfile:
    for key in extracted_data:
        json.dump({key: extracted_data[key]}, outfile)
        outfile.write('\n')
print(f"\nExtracted data saved to {output_file}")

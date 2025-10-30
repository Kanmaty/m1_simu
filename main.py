import numpy as np
import json


# --- (前のステップで作成したクラス定義) ---
class BaseStation:
    def __init__(self, pos=np.array([0, 0])):
        self.pos = pos


class SensorDevice:
    def __init__(self, device_id, pos):
        self.id = device_id
        self.pos = pos
        self.tx_power_watt = 10 ** (13 / 10) / 1000  # P_d (13 dBm)

        # X_d(t): 累積関連付け回数をトラック (セクション3.3)
        self.cumulative_associations = 0


class VehicleClusterRelay:
    def __init__(self, vcr_id, num_rvs, start_pos, velocity):
        self.id = vcr_id
        self.num_rvs = num_rvs
        self.pos = start_pos
        self.velocity = velocity
        self.rv_ids = tuple(range(num_rvs))

        # CHとRVの物理的な位置を仮定（簡単のためVCRと同じ位置）
        self.ch_pos = start_pos
        self.rv_positions = {rv_id: start_pos for rv_id in self.rv_ids}

        # CHの送信電力
        self.ch_tx_power_watt = 10 ** (13 / 10) / 1000  # P_v (13 dBm)

    def move(self, time_step):
        # (省略)
        pass


# --- (前のステップで作成した環境セットアップ関数) ---
def setup_environment(num_sensors, num_vcrs_list, num_channels):
    # (省略... 前のコードと同じ)
    bs = BaseStation()
    radius = 200
    angles = np.random.uniform(0, 2 * np.pi, num_sensors)
    distances = np.random.uniform(0, radius, num_sensors)
    sensor_positions = np.array(
        [distances * np.cos(angles), distances * np.sin(angles)]
    ).T
    sensors = {
        i: SensorDevice(i, pos) for i, pos in enumerate(sensor_positions)
    }  # 辞書型に変更

    vcrs = {}  # 辞書型に変更
    for i, num_rvs in enumerate(num_vcrs_list):
        vcr_pos = np.random.uniform(-radius, radius, 2)
        vcr_vel = np.random.uniform(-10, 10, 2)
        vcrs[i] = VehicleClusterRelay(i, num_rvs, vcr_pos, vcr_vel)

    channels = list(range(num_channels))
    return bs, sensors, vcrs, channels


# --- (前のステップで作成したチャネル割り当て関数) ---
def assign_channels_hardcoded(sensors, vcrs, channels):
    # (省略... 前のコードと同じ)
    allocations = {'d2i': {}, 'vcr_ch': {}, 'vcr_rv': {}}
    available_channels = channels.copy()

    # CH
    for vcr_id, vcr in vcrs.items():
        if not available_channels:
            break
        allocations['vcr_ch'][vcr_id] = available_channels.pop(0)

    # RV
    for vcr_id, vcr in vcrs.items():
        ch_channel = allocations['vcr_ch'].get(vcr_id)
        for rv_id in vcr.rv_ids:
            if not available_channels:
                break
            rv_channel = available_channels.pop(0)
            allocations['vcr_rv'][(vcr_id, rv_id)] = rv_channel

    # D2I
    for sensor_id in sensors.keys():
        if not available_channels:
            break
        allocations['d2i'][sensor_id] = available_channels.pop(0)

    return allocations


# シミュレーションパラメータ (表2  及び セクション3.2 )
SIM_PARAMS = {
    'channel_bandwidth': 500e3,  # W (500 KHz)
    'path_loss_exponent': 3.5,  # α
    'noise_watt': 10 ** (-117 / 10) / 1000,  # σ^2 (-117 dBm)
    # アンテナゲイン G (dBi) は簡単のため 1 (0 dBi) と仮定
    'antenna_gain': 1.0,
    # 最小伝送レート (Kb/s) -> (b/s)
    'r_min_bps': 500 * 1000,  # [cite: 628]
}

# --- ヘルパー関数 ---


def get_distance(pos1, pos2):
    """2点間のユークリッド距離を計算"""
    return np.linalg.norm(pos1 - pos2)


def get_path_loss(distance, alpha):
    """パスロス l^-α を計算"""
    # 距離が0または非常に近い場合のゼロ除算を避ける
    if distance < 1.0:
        distance = 1.0
    return distance ** (-alpha)


def calculate_link_capacity(signal_watt, interference_watt, noise_watt):
    """シャノン理論に基づきリンク容量 (b/s) を計算"""
    W = SIM_PARAMS['channel_bandwidth']
    sinr = signal_watt / (interference_watt + noise_watt)
    capacity_bps = W * np.log2(1 + sinr)
    return capacity_bps


# --- リンク容量計算 (干渉計算を含む) ---


def calculate_d2i_capacity(sensor_id, channel, all_sensors, all_vcrs, bs, allocations):
    """直接モード (D2I) のリンク容量を計算 (式2 )"""

    sensor = all_sensors[sensor_id]
    distance = get_distance(sensor.pos, bs.pos)
    path_loss = get_path_loss(distance, SIM_PARAMS['path_loss_exponent'])

    # 信号強度
    signal_watt = sensor.tx_power_watt * path_loss * SIM_PARAMS['antenna_gain']

    # 干渉合計 (式1 )
    interference_watt = 0

    # D2Vリンクからの干渉
    for (vcr_id, rv_id), rv_channel in allocations['vcr_rv'].items():
        if rv_channel == channel:
            # D2Vリンクにはセンサーが必要だが、このデモではどのセンサーが
            # RVに接続しているか未定。仮に、最も近いセンサーが接続すると仮定。
            # (ここでは簡単のため、ランダムなセンサーが干渉源と仮定)
            # ※ 本来は DA (Device Association) で決定される

            # 簡単化: このデモでは「RVに割り当てられたセンサー」を
            # all_sensors の先頭から順に仮定する
            # (この部分はアルゴリズム実装時にDAの結果で置き換える)
            try:
                # 仮の干渉源センサー
                interfering_sensor_id = list(all_sensors.keys())[
                    vcr_id * len(all_vcrs[vcr_id].rv_ids) + rv_id
                ]
                interfering_sensor = all_sensors[interfering_sensor_id]

                dist_interf = get_distance(interfering_sensor.pos, bs.pos)
                pl_interf = get_path_loss(dist_interf, SIM_PARAMS['path_loss_exponent'])
                interference_watt += (
                    interfering_sensor.tx_power_watt
                    * pl_interf
                    * SIM_PARAMS['antenna_gain']
                )
            except IndexError:
                pass  # 割り当てられるセンサーがいない

    return calculate_link_capacity(
        signal_watt, interference_watt, SIM_PARAMS['noise_watt']
    )


def calculate_d2v_capacity(
    sensor_id, vcr_id, rv_id, channel, all_sensors, all_vcrs, bs, allocations
):
    """間接モード (D2V) のリンク容量を計算 (式6 )"""

    sensor = all_sensors[sensor_id]
    vcr = all_vcrs[vcr_id]
    rv_pos = vcr.rv_positions[rv_id]

    distance = get_distance(sensor.pos, rv_pos)
    path_loss = get_path_loss(distance, SIM_PARAMS['path_loss_exponent'])

    # 信号強度
    signal_watt = sensor.tx_power_watt * path_loss * SIM_PARAMS['antenna_gain']

    # 干渉合計
    interference_watt = 0

    # 1. 他のD2Vリンクからの干渉 (式3 [cite: 181])
    for (other_vcr_id, other_rv_id), rv_channel in allocations['vcr_rv'].items():
        if rv_channel == channel and (other_vcr_id != vcr_id or other_rv_id != rv_id):
            # (D2Iと同様に、干渉源センサーを仮定)
            try:
                interfering_sensor_id = list(all_sensors.keys())[
                    other_vcr_id * len(all_vcrs[other_vcr_id].rv_ids) + other_rv_id
                ]
                interfering_sensor = all_sensors[interfering_sensor_id]

                dist_interf = get_distance(interfering_sensor.pos, rv_pos)
                pl_interf = get_path_loss(dist_interf, SIM_PARAMS['path_loss_exponent'])
                interference_watt += (
                    interfering_sensor.tx_power_watt
                    * pl_interf
                    * SIM_PARAMS['antenna_gain']
                )
            except IndexError:
                pass

    # 2. D2Iリンクからの干渉 (式4 [cite: 183])
    for sensor_d2i_id, d2i_channel in allocations['d2i'].items():
        if d2i_channel == channel:
            interfering_sensor = all_sensors[sensor_d2i_id]
            dist_interf = get_distance(interfering_sensor.pos, rv_pos)
            pl_interf = get_path_loss(dist_interf, SIM_PARAMS['path_loss_exponent'])
            interference_watt += (
                interfering_sensor.tx_power_watt
                * pl_interf
                * SIM_PARAMS['antenna_gain']
            )

    # 3. V2I (CH) リンクからの干渉 (式5 [cite: 232])
    for ch_vcr_id, ch_channel in allocations['vcr_ch'].items():
        if ch_channel == channel:
            interfering_vcr = all_vcrs[ch_vcr_id]
            dist_interf = get_distance(interfering_vcr.ch_pos, rv_pos)
            pl_interf = get_path_loss(dist_interf, SIM_PARAMS['path_loss_exponent'])
            interference_watt += (
                interfering_vcr.ch_tx_power_watt
                * pl_interf
                * SIM_PARAMS['antenna_gain']
            )

    return calculate_link_capacity(
        signal_watt, interference_watt, SIM_PARAMS['noise_watt']
    )


def calculate_v2i_capacity(vcr_id, channel, all_sensors, all_vcrs, bs, allocations):
    """間接モード (V2I) のリンク容量を計算 (式8 )"""

    vcr = all_vcrs[vcr_id]
    distance = get_distance(vcr.ch_pos, bs.pos)
    path_loss = get_path_loss(distance, SIM_PARAMS['path_loss_exponent'])

    # 信号強度
    signal_watt = vcr.ch_tx_power_watt * path_loss * SIM_PARAMS['antenna_gain']

    # 干渉合計 (式7 )
    interference_watt = 0

    # D2Vリンクからの干渉
    for (other_vcr_id, other_rv_id), rv_channel in allocations['vcr_rv'].items():
        if rv_channel == channel:
            # (D2Iと同様に、干渉源センサーを仮定)
            try:
                interfering_sensor_id = list(all_sensors.keys())[
                    other_vcr_id * len(all_vcrs[other_vcr_id].rv_ids) + other_rv_id
                ]
                interfering_sensor = all_sensors[interfering_sensor_id]

                dist_interf = get_distance(interfering_sensor.pos, bs.pos)
                pl_interf = get_path_loss(dist_interf, SIM_PARAMS['path_loss_exponent'])
                interference_watt += (
                    interfering_sensor.tx_power_watt
                    * pl_interf
                    * SIM_PARAMS['antenna_gain']
                )
            except IndexError:
                pass

    return calculate_link_capacity(
        signal_watt, interference_watt, SIM_PARAMS['noise_watt']
    )


# --- ステップ2: デバイスユーティリティ (式9) ---
def calculate_device_utility(sensor, has_associated_this_round=True):
    """
    センサーのデバイスユーティリティ ΔU_d(t) を計算する (式9)
    """
    if not has_associated_this_round:
        # 関連付けがなければユーティリティはゼロ [cite: 185]
        return 0.0

    # X_d(t-1): 前のピリオドまでの累積回数
    X_prev = sensor.cumulative_associations

    # log2(1 + 1 / (X_d(t-1) + 1))
    utility = np.log2(1.0 + 1.0 / (X_prev + 1.0))

    return utility


# --- ステップ3: デバイス優先度 (式15) ---
def calculate_da_priority(sensor, distance, comm_radius, t):
    """
    デバイス関連付けの優先度 O_d を計算する (式15)

    t: 現在の伝送ピリオド (t >= 1)
    distance: センサーとターゲット(RV or BS)の距離
    comm_radius: ターゲットの通信半径 (BSの場合は非常に大きい値)
    """

    # 1. ターゲットの通信範囲外か？
    if distance > comm_radius:
        # 式(15) の 3番目のケース: l_d,v(t) > r_n(t) [cite: 470]
        # 事実上、関連付け不可
        return -np.inf

    # 2. このラウンドで関連付いた場合のユーティリティを計算
    # (まだ関連付いていないので、X_prev は sensor.cumulative_associations)
    utility = calculate_device_utility(sensor, has_associated_this_round=True)

    # 3. 距離の正規化 (式15) [cite: 471]
    # (論文ではBSの半径 r_b で正規化しているが、
    #  ここではターゲットの半径 r_n(t) で正規化する)
    normalized_distance = distance / comm_radius
    distance_term = 1.0 - normalized_distance

    if t == 1:
        # 式(15) の 1番目のケース (t=1) [cite: 468]
        priority = utility + distance_term
    else:
        # 式(15) の 2番目のケース (t>1)
        # Lemma 1 に基づく重み [cite: 466, 468]
        # weight = log2(1 + 1/(t^2 - 1))
        # (t=1 の場合のゼロ除算を避けるため t>1 のみ)
        weight = np.log2(1.0 + 1.0 / (t**2 - 1.0))

        # 非常に小さい重み (ほぼ0) にならないよう下限を設ける
        if weight < 1e-9:
            weight = 1e-9

        priority = utility + (distance_term * weight)

    return priority


# --- ステップ4: DA スキーム (Scheme 3) の実装 ---
def device_association_scheme(sensors_dict, vcrs_dict, bs, t, channel_allocations):
    """
    DA (Device Association) スキーム (Scheme 3)

    戻り値:
    associations (dict): { 'd2i': [sensor_id, ...],
                           'd2v': { (vcr_id, rv_id): sensor_id } }
    """

    print(f'\n--- [伝送ピリオド {t}: DA スキーム実行] ---')

    associations = {'d2i': [], 'd2v': {}}

    # 関連付け可能なセンサーのリスト (コピー)
    available_sensors = list(sensors_dict.values())

    # 1. 間接モード (D2V) の関連付け (VCR -> RV)

    # VCRごとに処理 (Scheme 3, line 7) [cite: 492]
    for vcr_id, vcr in vcrs_dict.items():
        if not vcr.rv_ids:
            continue

        # VCRの通信半径 r_n(t) を計算 (式12)
        # (この計算は calculate_d2v_capacity に必要な干渉計算を伴うため複雑)
        # (ここでは仮に固定値 100m とします)
        comm_radius_vcr = 100.0  # 仮の値

        sensor_priorities = []

        # 関連付け可能な全センサーの優先度を計算 (Scheme 3, line 8) [cite: 498-499]
        for sensor in available_sensors:
            # (簡単のため、センサーとVCR(RV)の位置は同じと仮定)
            dist = get_distance(sensor.pos, vcr.pos)

            priority = calculate_da_priority(sensor, dist, comm_radius_vcr, t)

            if priority > -np.inf:
                sensor_priorities.append((priority, sensor))

        # 優先度で降順ソート (Scheme 3, line 9) [cite: 500]
        sensor_priorities.sort(key=lambda x: x[0], reverse=True)

        # 利用可能なRVの数だけ、上位のセンサーを割り当て (Scheme 3, line 10-12)
        num_rvs = len(vcr.rv_ids)
        assigned_count = 0

        for priority, sensor in sensor_priorities:
            if assigned_count >= num_rvs:
                break  # このVCRのRVはすべて埋まった

            # RVのIDを取得 (例: 0, 1, 2...)
            rv_id = vcr.rv_ids[assigned_count]

            # 関連付けを決定
            associations['d2v'][(vcr_id, rv_id)] = sensor.id

            # センサーを「関連付け可能」リストから削除
            available_sensors.remove(sensor)

            print(
                f'  D2V: Sensor {sensor.id} -> VCR {vcr_id} (RV {rv_id}) (Priority: {priority:.3f})'
            )
            assigned_count += 1

    # 2. 直接モード (D2I) の関連付け (Sensor -> BS)
    # (余ったチャネル数 と 余ったセンサー数 で決定)

    # D2Iで利用可能なチャネル数
    num_d2i_channels = len(channel_allocations['d2i'])

    sensor_priorities_d2i = []
    bs_comm_radius = 1000.0  # BSの半径は非常に大きいと仮定

    for sensor in available_sensors:
        dist = get_distance(sensor.pos, bs.pos)
        priority = calculate_da_priority(sensor, dist, bs_comm_radius, t)
        sensor_priorities_d2i.append((priority, sensor))

    # 優先度でソート
    sensor_priorities_d2i.sort(key=lambda x: x[0], reverse=True)

    # 利用可能なチャネル数だけ、上位のセンサーを割り当て
    for i in range(min(len(sensor_priorities_d2i), num_d2i_channels)):
        priority, sensor = sensor_priorities_d2i[i]
        associations['d2i'].append(sensor.id)
        print(f'  D2I: Sensor {sensor.id} -> BS (Priority: {priority:.3f})')

    return associations


# --- (シミュレーションのメインループを想定) ---
if __name__ == '__main__':
    # (環境セットアップ)
    NUM_SENSORS = 50
    VCR_CONFIG = [5, 5, 4]
    NUM_CHANNELS = 30
    bs, sensors_dict, vcrs_dict, channels = setup_environment(
        NUM_SENSORS, VCR_CONFIG, NUM_CHANNELS
    )

    # (チャネル割り当て: ハードコード)
    # D2I用に10チャネル、残りをVCR用と仮定
    allocations_fixed = {
        'd2i': channels[:10],  # 0-9
        'vcr_ch': {
            0: channels[10],  # VCR 0 (CH)
            1: channels[11],  # VCR 1 (CH)
            2: channels[12],  # VCR 2 (CH)
        },
        'vcr_rv': {
            # VCR 0 (RV 5台)
            (0, 0): channels[13],
            (0, 1): channels[14],
            (0, 2): channels[15],
            (0, 3): channels[16],
            (0, 4): channels[17],
            # VCR 1 (RV 5台)
            (1, 0): channels[18],
            (1, 1): channels[19],
            (1, 2): channels[20],
            (1, 3): channels[21],
            (1, 4): channels[22],
            # VCR 2 (RV 4台)
            (2, 0): channels[23],
            (2, 1): channels[24],
            (2, 2): channels[25],
            (2, 3): channels[26],
            # 残りチャネル: channels[27], channels[28], channels[29] (未使用)
        },
    }
    # === シミュレーションループ ===
    TOTAL_PERIODS = 10
    for t in range(1, TOTAL_PERIODS + 1):
        # 1. DAスキームで、どのセンサーが通信するか決定
        # (チャネル割り当て自体はまだ固定)
        da_results = device_association_scheme(
            sensors_dict, vcrs_dict, bs, t, allocations_fixed
        )

        # 2. ユーティリティの更新と累積回数の加算
        total_utility_this_round = 0

        # D2Vで関連付いたセンサー
        for sensor_id in da_results['d2v'].values():
            sensor = sensors_dict[sensor_id]
            total_utility_this_round += calculate_device_utility(sensor)
            sensor.cumulative_associations += 1  # X_d(t) = X_d(t-1) + 1 [cite: 197]

        # D2Iで関連付いたセンサー
        for sensor_id in da_results['d2i']:
            sensor = sensors_dict[sensor_id]
            total_utility_this_round += calculate_device_utility(sensor)
            sensor.cumulative_associations += 1

        print(
            f'  ピリオド {t} 完了。 合計ユーティリティ: {total_utility_this_round:.3f}'
        )

        # 3. VCRの移動 (次のピリオドの準備)
        for vcr in vcrs_dict.values():
            vcr.move(time_step=1.0)  # 1秒分移動

    print('\n--- シミュレーション完了 ---')
    for sensor in sensors_dict.values():
        print(f'Sensor {sensor.id}: 最終累積回数 = {sensor.cumulative_associations}')

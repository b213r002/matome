import pandas as pd
import numpy as np

# MatchLoaderクラスの定義
class MatchLoader:
    # CSVから試合内容を読み込むメソッド
    def load_match_info(self):
        # CSVファイルの読み込み (ヘッダーなし)
        csv_data = pd.read_csv("W.csv", header=None)

        # 最後の列（ムーブシーケンス）だけを抽出
        move_sequences = csv_data.iloc[:, -1]  # 最後の列を抽出

        # 正規表現を使って2文字ずつ切り出す
        extract_one_hand = move_sequences.str.extractall(r'(..)')

        # Indexを再構成して、1行1手の表にする
        # 試合の切り替わり判定のためgame_idも残しておく
        one_hand_df = extract_one_hand.reset_index().rename(columns={"level_0": "tournamentId", "level_1": "match", 0: "move_str"})

        # アルファベットを数字に変換するテーブル
        conv_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
        
        # 1手を数値に変換する
        one_hand_df["move"] = one_hand_df.apply(lambda x: self.convert_move(x["move_str"], conv_table), axis=1)

        return one_hand_df

    # 1手を数値に変換するメソッド
    def convert_move(self, v, conv_table):
        l = conv_table[v[:1]]  # 列の値を変換する
        r = int(v[1:])         # 行の値を変換する
        return np.array([l - 1, r - 1], dtype='int8')

# メイン処理
if __name__ == "__main__":
    match_loader = MatchLoader()
    one_hand_df = match_loader.load_match_info()
    print(one_hand_df.head())  # 結果を確認するために表示

class ReversiProcessor:
    def process_tournament(self, df):
        # 試合が切り替わる盤面リセット
        if df["tournamentId"] != self.now_tournament_id:
            self.table_info = [0] * 100  # 10x10のボードで外枠は無視する
            # 初期配置
            self.table_info[44] = 2
            self.table_info[45] = 1
            self.table_info[54] = 1
            self.table_info[55] = 2
            self.turn_color = 1
            self.now_tournament_id = df["tournamentId"]
        else:
            self.turn_color = 1 if self.turn_color == 2 else 2

        # 置ける箇所がなければパスする
        if len(self.GetCanPutPos(self.turn_color, self.table_info)) == 0:
            self.turn_color = 1 if self.turn_color == 2 else 2

        # 配置場所
        put_pos = df["move"]

        # 訓練用データ追加
        self.record_training_data(put_pos)

        # 盤面更新
        put_index = put_pos[0] + (put_pos[1]) * 10
        self.PutStone(put_index, self.turn_color, self.table_info)

    def record_training_data(self, put_pos):
        # ボード情報を自分と敵のものに分ける
        my_board_info = np.zeros(shape=(8, 8), dtype="int8")
        enemy_board_info = np.zeros(shape=(8, 8), dtype="int8")

        for i in range(11, 89):  # 10x10のボードの内側(8x8部分)を処理
            if i % 10 == 0 or i % 10 == 9:
                continue  # 余分な枠をスキップ

            board_x = (i % 10) - 1
            board_y = (i // 10) - 1

            if self.table_info[i] == 1:
                my_board_info[board_y][board_x] = 1
            elif self.table_info[i] == 2:
                enemy_board_info[board_y][board_x] = 1

        move_one_hot = np.zeros(shape=(8, 8), dtype='int8')
        move_one_hot[put_pos[1]][put_pos[0]] = 1

        # 訓練データを記録
        if self.turn_color == 1:
            self.my_board_infos.append(np.array([my_board_info.copy(), enemy_board_info.copy()], dtype="int8"))
            self.my_put_pos.append(move_one_hot)
        else:
            self.enemy_board_infos.append(np.array([enemy_board_info.copy(), my_board_info.copy()], dtype="int8"))
            self.enemy_put_pos.append(move_one_hot)

    # ダミーのメソッド(詳細な実装が必要)
    def GetCanPutPos(self, turn_color, table_info):
        # 置ける場所をリストとして返す
        # 実際にはこの部分でルールに従った処理が必要
        return [pos for pos in range(100) if table_info[pos] == 0]

    def PutStone(self, put_index, turn_color, table_info):
        # 石を置く処理
        # 実際にはこの部分でルールに従った処理が必要
        table_info[put_index] = turn_color


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    class Bias(keras.layers.Layer):
        def __init__(self, input_shape):
            super(Bias, self).__init__()
            self.W = tf.Variable(initial_value=tf.zeros(input_shape), trainable=True)

        def call(self, inputs):
            return inputs + self.W

    model = keras.Sequential()
    model.add(layers.Permute((2, 3, 1), input_shape=(2, 8, 8)))
    for _ in range(11):  # 畳み込み層を11回追加
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.Conv2D(1, kernel_size=1, use_bias=False))
    model.add(layers.Flatten())
    model.add(Bias((64,)))  # 修正点
    model.add(layers.Activation('softmax'))

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('モデルは正常')  # 修正点
    return model

def training(self):
    x_train = np.concatenate([self.my_board_infos, self.enemy_board_infos])
    y_train_tmp = np.concatenate([self.my_put_pos, self.enemy_put_pos])
 
    # 教師データをサイズ64の1次元配列に変換
    y_train = y_train_tmp.reshape(-1, 64)
 
    try:
        # 学習を開始
        model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)
    except KeyboardInterrupt:
        # 学習中に途中で中断された場合に途中結果を出力
        model.save('saved_model_reversi/my_model_interrupt')
        print('Output saved')
        return
     
    # 学習が終了したら指定パスに結果を出力
    model.save('saved_model_reversi/my_model')
    print('complete')

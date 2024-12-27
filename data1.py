import pandas as pd
import numpy as np
import random
import time

# MatchLoader: WTHORから試合データを読み込む
class MatchLoader:
    def __init__(self, csv_file):
        self.csv_file = csv_file
    
    def load_match_info(self):
        """CSVファイルから試合内容を読み込み、新しいDataFrameを作成"""
        try:
            csv_data = pd.read_csv(self.csv_file, header=None)
            move_sequences = csv_data.iloc[:, -1]
            extract_one_hand = move_sequences.str.extractall(r'(..)')

            one_hand_df = extract_one_hand.reset_index().rename(
                columns={"level_0": "tournamentId", "level_1": "match", 0: "move_str"}
            )

            conv_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
            one_hand_df["move"] = one_hand_df["move_str"].apply(lambda x: self.convert_move(x, conv_table))
            
            # 新しいDataFrameを作成し、不要な列を削除
            new_df = one_hand_df.drop(columns=['move_str'])
            return new_df
        except Exception as e:
            print(f"試合情報の読み込み中にエラーが発生しました: {e}")
            return None

    def convert_move(self, move_str, conv_table):
        """1手を数値に変換する"""
        col = conv_table[move_str[0]]
        row = int(move_str[1])
        return np.array([col - 1, row - 1], dtype='int8')

# メイン処理
if __name__ == "__main__":
    # W.csvを読み込み、新しいDataFrameを作成
    match_loader = MatchLoader("W.csv")
    new_df = match_loader.load_match_info()

    # W1.csvに書き出し
    new_df.to_csv("W1.csv", index=False)
    print("W1.csvファイルが作成されました。")
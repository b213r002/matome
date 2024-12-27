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



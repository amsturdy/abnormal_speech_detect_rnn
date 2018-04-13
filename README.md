# rare_sound_event_detection_rnn
# 说明：
使用cnn+GRU对每帧音频信号进行分类，从而达到检测的目的

# 项目目录结构为：
 |-------rare_sound_event_detection_rnn（整个项目目录）
 | |---------data（数据目录）
 | |---------src（存放所有程序目录）
 | |  |-----------audio_event_detect.py（搭建模型、模型训练、模型预测）
 | |  |-----------config.py（实验参数设置）
 | |  |-----------data_generator.py（为模型生成batch_size的训练数据）
 | |  |-----------evaluate.py（评测检测效果）
 | |  |-----------prepare_data.py（特征提取，数据预处理、打包、归一化等）
 | |  |-----------run.sh（整个项目流程）
 | |  |-----------synthesizer（存放合成语音数据所有程序的目录）
 | |  |   |------------------config.py（合成语音数据的一些参数设置）
 | |  |   |------------------generate_mixtures.py（合成语音数据）
 | |  |   |------------------requirements.txt（合成语音数据所需的依赖）

# 使用：
在项目主目录下运行 sh ./src/run.sh，就能完成从数据合成到模型训练和预测的所有步骤。

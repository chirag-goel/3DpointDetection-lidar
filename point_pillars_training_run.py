import os
import tensorflow as tf
from glob import glob

from config import Parameters
from loss import PointPillarNetworkLoss
from network import build_point_pillar_graph
from processors import SimpleDataGenerator
from readers import KittiDataReader

DATA_ROOT = "/Users/chirag/Documents/Projects/682/PointPillars/data"  # TODO make main arg

TEST_ROOT = "/Users/chirag/Documents/Projects/682/PointPillars/data/testing"  # TODO make main arg

if __name__ == "__main__":

    params = Parameters()

    pillar_net = build_point_pillar_graph(params)

    loss = PointPillarNetworkLoss(params)

    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate, decay=params.decay_rate)

    pillar_net.compile(optimizer, loss=loss.losses(),metrics=['accuracy'])

    log_dir = "./logs"
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, "model.h5"), save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.8 if ((epoch % 15 == 0) and (epoch != 0)) else lr, verbose=True),
        tf.keras.callbacks.EarlyStopping(patience=20),
    ]

    data_reader = KittiDataReader()

    lidar_files = sorted(glob(os.path.join(DATA_ROOT, "velodyne", "*.bin")))
    label_files = sorted(glob(os.path.join(DATA_ROOT, "label", "*.txt")))
    calibration_files = sorted(glob(os.path.join(DATA_ROOT, "calib", "*.txt")))


    lidar_files = lidar_files[:50]
    label_files = label_files[:50]
    calibration_files = calibration_files[:50]


    valVal = (int)(len(lidar_files)*0.7)

    testVal = (int)(len(lidar_files)*0.9)

    print(valVal)

    print(testVal)

    lidar_files_val = lidar_files[valVal:testVal]
    label_files_val = label_files[valVal:testVal]
    calibration_files_val = calibration_files[valVal:testVal]

    lidar_files_test = lidar_files[testVal:]
    label_files_test = label_files[testVal:]
    calibration_files_test = calibration_files[testVal:]

    lidar_files = lidar_files[:valVal]
    label_files = label_files[:valVal]
    calibration_files = calibration_files[:valVal]

    print(lidar_files)
    print(lidar_files_val)
    print(lidar_files_test)

    training_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files, label_files, calibration_files)

    validation_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files_val, label_files_val, calibration_files_val)

    test_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files_test, label_files_test, calibration_files_test)


    try:
        pillar_net.fit_generator(training_gen,
                                 len(training_gen),
                                 callbacks=callbacks,
                                 use_multiprocessing=True,
                                 epochs=int(params.total_training_epochs),
                                 workers=6,
                                 validation_data = validation_gen,
                                 validation_steps = len(validation_gen))


        print("Test-----")
        results = pillar_net.evaluate_generator(test_gen,verbose=1)



        # print('test loss, test acc:', results)
        print("Predictions-----")
        print('\n# Generate predictions for 3 samples')
        predictions = pillar_net.predict_generator(test_gen,verbose=1)
        print('predictions')
        print(predictions)


    except KeyboardInterrupt:
        pillar_net.save(os.path.join(log_dir, "interrupted.h5"))
        session = tf.keras.backend.get_session()
        session.close()

import inspect

init_str = """
import pandas as pd
import os

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

"""

with open('run_model_on_gpu.py', 'w') as f:

    f.write(init_str)
    f.write('\n\n')

    for fn_name in [load_train, load_test, create_model, train_model]:

        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')

    f.write("""

# Main execution block
if __name__ == '__main__':
    # Define the path to your dataset
    path = '/content'

    # Load training and testing data
    train_gen = load_train(path)
    test_gen = load_test(path)

    # Create the model
    model = create_model(input_shape=(224, 224, 3))

    # Train the model
    train_model(model, train_gen, test_gen)

    # Evaluate the model on the test data (optional, but good practice)
    print('Evaluando el modelo en el conjunto de prueba:')
    metrics = model.evaluate(test_gen, verbose=2)
    print(f'Test MAE: {metrics[1]:.3f}')

""")

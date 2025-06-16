import numpy as np
import pandas as pd

def generate_sample_data(n_samples=1000, data_type="image"):
    if data_type == "image":
        X = np.random.rand(n_samples, 3, 32, 32)
        y = np.random.randint(0, 10, n_samples)
        df = pd.DataFrame(X.reshape(n_samples, -1))
        df['label'] = y
        df.to_csv("sample_image_data.csv", index=False)
    elif data_type == "sequence":
        X = np.random.rand(n_samples, 10, 20)
        y = np.random.randint(0, 10, n_samples)
        df = pd.DataFrame(X.reshape(n_samples, -1))
        df['label'] = y
        df.to_csv("sample_sequence_data.csv", index=False)
    elif data_type == "tabular":
        X = np.random.rand(n_samples, 50)
        y = np.random.randint(0, 10, n_samples)
        df = pd.DataFrame(X)
        df['label'] = y
        df.to_csv("sample_tabular_data.csv", index=False)

if __name__ == "__main__":
    generate_sample_data(data_type="image")
    generate_sample_data(data_type="sequence")
    generate_sample_data(data_type="tabular")
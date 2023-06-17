import numpy as np

class DataLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.m_data = data.shape[0]
        self.size = int(self.m_data/batch_size)
       
    def __iter__(self):
        np.random.shuffle(self.data)
        data = np.split(self.data, self.size)
        self.iter_data = iter(data)
        return self.iter_data
    
    def __next__(self):
        return next(self.iter_data)

if __name__ == "__main__":
    data = np.random.randint(1, 10, (10, 3))
    dataloader = DataLoader(data, batch_size=5)
    for i in dataloader:
        print(i)
        print()
        
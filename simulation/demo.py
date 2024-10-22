if __name__ == '__main__':
    import numpy as np
    data = np.load('simulation_data/cube-task-effects-img.npy', allow_pickle=True)
    from PIL import Image
    img = Image.fromarray(data[0])
    img.show()

import numpy as np
from matplotlib import pyplot
from PIL import Image

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1) **2)

def bmp_to_boolean_array(image_path):
    # Open the BMP image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Get dimensions
    width, height = img.size
    print(f"Width: {width}, Height: {height}")
    
    # Convert to NumPy array
    img_array = np.array(img)
    
    # Normalize to boolean: white (255) -> False, black (0) -> True
    bool_array = img_array == 255
    
    return width, height, bool_array

def main():
    # process image
    image_path = "Obstacle.bmp"
    Nx, Ny, Obstacle = bmp_to_boolean_array(image_path)
    tau = 0.53
    Nt = 3000
    Plot_every = 25

    # Speeds and weights
    Nl = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # Innitial Conditions
    F = np.ones((Ny, Nx, Nl), dtype=np.float64) + .01 * np.random.randn(Ny, Nx, Nl)
    F[:, :, 3] = 1.25

    # Origin
    Origin = [2, Ny//2]
    Width = 10
    
    # Set initial weight for F to 2.5 only between the two obstacles
    for y in range(Ny//2 - Width//2, Ny//2 + Width//2):
        for x in range(Origin[0], Nx//4 ):
            F[y, x, 3] = 3
    
    # Run
    for epoch in range(Nt):

        # Dampen end wall
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]
        
        F[0, :, [8, 1, 2]] = F[1, :, [8, 1, 2]]
        F[-1, :, [4, 5, 6]] = F[-2, :, [4, 5, 6]]
        
        # Stream
        for i, cx, cy in zip(range(Nl), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)
        
        # Colisions Effect
        bndryF = F[Obstacle, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Fluid Property Variables
        Rho = np.sum(F, 2)
        Ux = np.sum(F * cxs, 2) / Rho
        Uy = np.sum(F * cys, 2) / Rho

        # Obstacle Property Variables
        F[Obstacle, :] = bndryF
        Ux[Obstacle] = 0
        Uy[Obstacle] = 0

        # Colisions Calculation
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(Nl), cxs, cys, weights):
            Feq [:, :, i] = np.clip(Rho * w * (
                1 + (3 * (cx * Ux + cy * Uy )) + (9 * ((cx * Ux + cy * Uy ) ** 2) / 2) - (3 * (Ux ** 2 + Uy ** 2) / 2)
            ),0, 10)
        F = F + -(1/tau) * (F - Feq)

        # Plot
        if(epoch % Plot_every == 0):
            dfydx = Ux[2:, 1:-1] - Ux[0:-2 , 1:-1]
            dfxdy = Uy[1:-1, 2:] - Uy[1:-1, 0:-2]
            curl = dfydx - dfxdy
            pyplot.imshow(curl)
            #pyplot.imshow(np.sqrt(Ux ** 2 + Uy ** 2))
            pyplot.pause(.01)
            pyplot.cla()
        

if __name__ == "__main__":
    main()

import numpy as np
from matplotlib import pyplot

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1) **2)

def main():
    Nx = 100
    Ny = 100
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
    F[:, :, 3] = 1

    # Obstacle
    Obstacle = np.full((Ny, Nx), False)
    Origin = [5, Ny//2]
    Radius = 10
    Barel_wall = 2
    flash_supressor_gap = Radius//2
    for y in range(0, Ny):
        for x in range(0, Nx):
            Obstacle[y,x] = (distance(Origin[0], Origin[1], x, y) > Radius and distance(Origin[0], Origin[1], x, y) < Radius + Barel_wall) and \
                            (y > Origin[1] + flash_supressor_gap or y < Origin[1] - flash_supressor_gap)
            pass
    
    # Set initial weight for F to 2.5 only between the two obstacles
    for y in range(Origin[1] - Radius, Origin[1] + Radius):
        for x in range(5, 10):
            if y > (Origin[1] - Radius) and y < (Origin[1] + Radius):
                F[y, x, 3] = 2.5
    
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
            Feq [:, :, i] = (Rho * w * (
                1 + 3 * (cx * Ux + cy * Uy ) + 9 * (cx * Ux + cy * Uy ) ** 2 / 2 - 3 * (Ux ** 2 + Uy ** 2) / 2
            ))
        F = F + -(1/tau) * (F - Feq)

        # Plot
        if(epoch % Plot_every == 0):
            #dfydx = Ux[2:, 1:-1] - Ux[0:-2 , 1:-1]
            #dfxdy = Uy[1:-1, 2:] - Uy[1:-1, 0:-2]
            #curl = dfydx - dfxdy
            #pyplot.imshow(curl)
            pyplot.imshow(np.sqrt(Ux ** 2 + Uy ** 2))
            pyplot.pause(.01)
            pyplot.cla()
        

if __name__ == "__main__":
    main()

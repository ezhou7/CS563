from tracking.twodim.filter import ParticleFilter2D


NUM_PARTICLES = 100


def main():
    pfilter = ParticleFilter2D(NUM_PARTICLES)
    pfilter.track_cells_in_2d()


if __name__ == "__main__":
    main()

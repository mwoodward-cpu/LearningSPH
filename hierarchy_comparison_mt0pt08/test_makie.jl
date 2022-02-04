# using GLMakie
# using Makie
# AbstractPlotting.inline!(false)
#
# r = LinRange(-1, 1, 100)
# cube = [(x.^2 + y.^2 + z.^2) for x = r, y = r, z = r]
# cube_with_holes = cube .* (cube .> 1.4)
#
# scene = Makie.volume(cube_with_holes, algorithm = :iso, isorange = 0.05, isovalue = 1.7)
# # Makie.save("./check.png", scene)
# # display(scene)

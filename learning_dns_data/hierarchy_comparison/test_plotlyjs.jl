using FileIO, Images, HTTP
using PlotlyJS

url = "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif"

download(url, "attention-mri.tif")
img = load("attention-mri.tif");
GRAY = permutedims(channelview(img), (2,1,3));

r, c = size(GRAY[1, :, :])
n_slices = size(GRAY, 1)
height = (n_slices-1) / 10
grid = LinRange(0, height, n_slices)
slice_step = grid[2] - grid[1]


pl_bone=[[0.0, "rgb(0, 0, 0)"],
         [0.1, "rgb(21, 21, 30)"],
         [0.2, "rgb(44, 44, 62)"],
         [0.3, "rgb(66, 66, 92)"],
         [0.4, "rgb(89, 92, 121)"],
         [0.5, "rgb(112, 123, 143)"],
         [0.6, "rgb(133, 153, 165)"],
         [0.7, "rgb(156, 184, 188)"],
         [0.8, "rgb(185, 210, 210)"],
         [0.9, "rgb(220, 233, 233)"],
         [1.0, "rgb(255, 255, 255)"]];

initial_slice = PlotlyJS.surface(
                     z=height*ones(r,c),
                     surfacecolor=GRAY[end, end:-1:1, :],
                     colorscale=pl_bone,
                     reversescale=true, #commenting out this line we get  darker slices
                     showscale=false)


frames  = Vector{PlotlyFrame}(undef, n_slices)
for k in 1:n_slices
    frames[k] = PlotlyJS.frame(data=[attr(
                                 z=(height-(k-1)*slice_step)*ones(r,c),
                                 surfacecolor=GRAY[end-(k-1), end:-1:1, :])],
                                 name="fr$k",
                                 traces=[0])
end

sliders = [attr(steps = [attr(method= "animate",
                              args= [["fr$k"],
                              attr(mode= "immediate",
                                   frame= attr(duration=40, redraw= true),
                                   transition=attr(duration= 0))
                                 ],
                              label="$k"
                             ) for k in 1:n_slices],
                active=17,
                transition= attr(duration= 0 ),
                x=0, # slider starting position
                y=0,
                currentvalue=attr(font=attr(size=12),
                                  prefix="slice: ",
                                  visible=true,
                                  xanchor= "center"
                                 ),
               len=1.0) #slider length
           ];
layout = PlotlyJS.Layout(title_text="Head Scanning", title_x=0.5,
                width=600,
                height=600,
                scene_zaxis_range= [-0.1, 6.8],
                sliders=sliders,
            )
pl= PlotlyJS.Plot(initial_slice, layout, frames)

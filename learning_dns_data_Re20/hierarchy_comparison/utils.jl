function make_dir(path)
        if isdir(path) == true
               println("directory already exists")
           else mkdir(path)
        end
end

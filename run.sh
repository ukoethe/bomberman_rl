docker build ./
gnome-terminal -e 'sh -c "sleep 3; remmina -c vnc://localhost:5900"' &
docker run --rm -p 5900:5900 -v $PWD:/home/bomberman bomberman x11vnc -forever -usepw -create
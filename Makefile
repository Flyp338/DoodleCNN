CC = gcc
CFLAGS = -Wall -Wextra -I c/include
SRC = c/src/tensor.c c/src/model_loader.c c/src/layers.c

all: doodle_app

doodle_app: $(SRC) c/src/main.c
	$(CC) $(CFLAGS) $^ -o doodle_app.exe

run: doodle_app
	.\doodle_app.exe

clean:
	rm -f *.exe
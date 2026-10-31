# Compiling a graph for the NCS1 (NCSDK v1 format)

Intel's v1 compiler (`mvNCCompile.py`) is plain Python; it needs Caffe (or TensorFlow 1.x) to parse the net.
Run it in a throw-away container with no network. Nothing here is run from inside the downloaded tree.

```bash
W=$HOME/ncs1-work                       # any new directory
tools/ncs1-compile/fetch.sh $W/toolkit  # downloads + verifies the toolkit (empty dir required)
docker build -t ncs1-compiler:local tools/ncs1-compile
# model dir must hold NAME.prototxt (batch size 1!) and NAME.caffemodel
docker run --rm --network none \
  -v $W/toolkit/x/NCSDK-1.12.01.01/ncsdk-x86_64/tk:/tk:ro \
  -v $PWD/tools/ncs1-compile/stub:/stub:ro \
  -v $W/models:/models:ro -v $PWD/tools/ncs1-compile/compile.py:/compile.py:ro \
  -v $W/out:/out -w /out ncs1-compiler:local \
  python3 -I /compile.py /models/NAME.prototxt -s 12 -o /out/NAME.graph
```

`stub/mvnc` replaces Intel's host Python module: the compile path never talks to a device, and we
do not use Intel's host library at all.

`compare_caffe_cpu.py` is the CPU reference check (Caffe fp32 vs the stick output); see docs/USING_THE_STICK.md.

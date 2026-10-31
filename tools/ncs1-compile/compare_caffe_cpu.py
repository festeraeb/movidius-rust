import sys, numpy as np
sys.path.insert(0, "/stub")
import caffe
caffe.set_mode_cpu()
x = np.fromfile("/cmp/mv_in.f32", dtype="<f4").reshape(227, 227, 3)   # HWC, already preprocessed (fp16-rounded)
stick = np.fromfile("/cmp/mv_out.f32", dtype="<f4")
net = caffe.Net("/models/squeezenet11.prototxt", "/models/squeezenet11.caffemodel", caffe.TEST)
net.blobs["data"].data[...] = x.transpose(2, 0, 1)[None]
cpu = net.forward()["prob"].reshape(-1).astype(np.float32)
labels = open("/cmp/synset_words.txt").read().splitlines()
def top(v): return list(np.argsort(-v)[:5])
print("CPU (Caffe fp32) top5:")
for i in top(cpu): print("  %4d %.5f %s" % (i, cpu[i], labels[i]))
print("STICK top5:")
for i in top(stick): print("  %4d %.5f %s" % (i, stick[i], labels[i]))
print("same top5 order:", top(cpu) == top(stick), " same top1:", top(cpu)[0] == top(stick)[0])
d = np.abs(cpu - stick)
print("max abs diff: %.6f at class %d (cpu %.5f stick %.5f)" % (d.max(), d.argmax(), cpu[d.argmax()], stick[d.argmax()]))
print("mean abs diff: %.7f  sum cpu %.5f sum stick %.5f" % (d.mean(), cpu.sum(), stick.sum()))

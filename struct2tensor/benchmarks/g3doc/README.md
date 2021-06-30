# Benchmarks Results

<!--*
freshness: { owner: 'andylou' reviewed: '2021-04-02' }
*-->

The purpose of these benchmarks is to help users evaluate Prensor's performance.
We compare its performance against the traditional TensorFlow container format,
tf.Examples. The protos used in these benchmarks are contrived to illustrate
extreme cases.

These benchmarks measure some of struct2tensor' most used operations, as well as
its "end to end" timings. Where "end to end" is from a tensor of serialized
string proto (arbitrary proto or tf.Example) to Tensors that can be used to
train or do inference.

Note: there is no measurement of I/O at all in any of these benchmarks. We can
perhaps see even greater gains when using a columnar storage with prensors.

[TOC]

## Prensor Operations

### Projecting Fields

These benchmarks measures the time it takes to project fields in a Prensor tree.

We found that project scales with the nestedness of the submessages. This is due
to decoding nested fields being more expensive than flat fields. The prensor
operations (project, promote, broadcast, and reroot) are near trivial compared
to the actual decoding of the proto.

Project 1 int field from a flat proto:

![image](images/project_flat_1.png, "This tree represents a protobuf. The leaf nodes are repeated fields.")
<!-- source: https://docs.google.com/drawings/d/1AJn-Rr5Q39l_Sw-7uSqhtq6KOC_JCMEiFosfG8A5zws/edit -->

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.322     | 0.119     | 0.340    | 4.761
2          | 0.318     | 0.147     | 0.300    | 4.606
4          | 0.310     | 0.037     | 0.320    | 4.688
8          | 0.320     | 0.071     | 0.330    | 4.726
16         | 0.328     | 0.182     | 0.340    | 4.761
32         | 0.322     | 0.057     | 0.330    | 4.726
64         | 0.331     | 0.058     | 0.330    | 4.726
128        | 0.353     | 0.052     | 0.360    | 4.824
256        | 0.394     | 0.055     | 0.390    | 4.902
512        | 0.469     | 0.063     | 0.480    | 5.021
1024       | 0.615     | 0.059     | 0.630    | 4.852

Project 5 int field from a flat proto:

![image](images/project_flat_5.png, "Project all fields.")
<!-- source: https://docs.google.com/drawings/d/1OqFr-1mQLrwBsS__TwT6Esv_uoi1bjzg9GKXZYuZr98/edit -->

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.315     | 0.143     | 0.330    | 4.726
2          | 0.311     | 0.043     | 0.320    | 4.688
4          | 0.313     | 0.047     | 0.320    | 4.688
8          | 0.315     | 0.048     | 0.320    | 4.688
16         | 0.319     | 0.054     | 0.330    | 4.726
32         | 0.325     | 0.043     | 0.340    | 4.761
64         | 0.339     | 0.070     | 0.330    | 4.726
128        | 0.366     | 0.048     | 0.380    | 4.878
256        | 0.420     | 0.050     | 0.470    | 5.016
512        | 0.519     | 0.063     | 0.470    | 5.016
1024       | 0.711     | 0.073     | 0.750    | 4.578

Project 1 int field from a deep proto:

This graph represents a protobuf. The leaf nodes are repeated fields, and the
internal nodes represent nested submessages.

![image](images/project_deep_1.png, "This tree represents a nested protobuf. Each leaf node is a proto field, and each internal node is a nested submessage.")
<!-- source: https://docs.google.com/drawings/d/19CXMEepBJRCICg1eCYUCDEyAb5A4tkclpbkIHd4UV4A/edit -->

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.322     | 0.141     | 0.340    | 4.761
2          | 0.323     | 0.090     | 0.350    | 4.794
4          | 0.320     | 0.131     | 0.320    | 4.688
8          | 0.319     | 0.123     | 0.320    | 4.688
16         | 0.328     | 0.100     | 0.320    | 4.688
32         | 0.323     | 0.154     | 0.310    | 4.648
64         | 0.330     | 0.145     | 0.340    | 4.761
128        | 0.346     | 0.101     | 0.340    | 4.761
256        | 0.373     | 0.046     | 0.390    | 4.902
512        | 0.432     | 0.058     | 0.440    | 4.989
1024       | 0.538     | 0.051     | 0.540    | 5.009

Project 5 int field from a deep proto:

![image](images/project_deep_5.png, "Project a field of depth 5.")
<!-- source: https://docs.google.com/drawings/d/1W7O6YZ2svXQ25BpdaUUDA0nLwfTTk8CgFFCPXvU1Po8/edit -->

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.317     | 0.123     | 0.330    | 4.726
2          | 0.317     | 0.036     | 0.320    | 4.688
4          | 0.317     | 0.024     | 0.340    | 4.761
8          | 0.323     | 0.136     | 0.340    | 4.761
16         | 0.324     | 0.066     | 0.320    | 4.688
32         | 0.332     | 0.028     | 0.340    | 4.761
64         | 0.351     | 0.052     | 0.340    | 4.761
128        | 0.389     | 0.035     | 0.410    | 4.943
256        | 0.466     | 0.114     | 0.500    | 5.025
512        | 0.607     | 0.074     | 0.610    | 4.902
1024       | 0.885     | 0.064     | 0.930    | 3.258

### Promoting Fields

The findings here are similar to the findings in project. Promote does not add
much cpu cost.

Promote an int field to depth 1 and project it:

![image](images/promote_depth_1.png, "Promote a field to depth 1 and project it.")
<!-- source: https://docs.google.com/drawings/d/13mB5j0PV8Bb4ALeXBVCaXAlMvqJtOhBh16r366-rqQY/edit -->

Batch Size | Wall Time    | Wall Time     | CPU Time | CPU Time
---------- | ------------ | ------------- | -------- | -----------
           | avg (ms)     | (stdev)       | avg (ms) | (stdev)
1          | 0.3518498302 | 0.1188202848  | 0.35     | 4.793724854
2          | 0.3557216492 | 0.1009445213  | 0.35     | 4.793724854
4          | 0.3582275826 | 0.1145162095  | 0.36     | 4.824181513
8          | 0.3468946675 | 0.1553422029  | 0.35     | 4.793724854
16         | 0.343196976  | 0.02561573622 | 0.37     | 4.852365871
32         | 0.3489765655 | 0.08888674588 | 0.36     | 4.824181513
64         | 0.3577148998 | 0.03166332931 | 0.35     | 4.793724854
128        | 0.3794417465 | 0.02612066547 | 0.41     | 4.943110704
256        | 0.4227451435 | 0.03535880411 | 0.45     | 5
512        | 0.5022811694 | 0.03165055746 | 0.53     | 5.403889359
1024       | 0.6583163114 | 0.04877839704 | 0.74     | 5.04925237

Promote an int field to depth 4 and project it:

![image](images/promote_depth_4.png, "Promote a field to depth 4 and project it.")
<!-- source: https://docs.google.com/drawings/d/1RMP7RScrZyFU61XjztmMol7ROsjh-qQ-0_WchcfJY-A/edit -->

Batch Size | Wall Time    | Wall Time     | CPU Time | CPU Time
---------- | ------------ | ------------- | -------- | -----------
           | avg (ms)     | (stdev)       | avg (ms) | (stdev)
1          | 0.3421712951 | 0.1137565235  | 0.36     | 4.824181513
2          | 0.3411100265 | 0.03153159862 | 0.35     | 4.793724854
4          | 0.3419795991 | 0.03430791486 | 0.35     | 4.793724854
8          | 0.3459784086 | 0.1218973419  | 0.34     | 4.760952286
16         | 0.3500788859 | 0.100308702   | 0.36     | 4.824181513
32         | 0.3584371894 | 0.02694506838 | 0.33     | 4.725815626
64         | 0.376711451  | 0.03072844429 | 0.38     | 4.878317312
128        | 0.4139752886 | 0.03082536188 | 0.41     | 4.943110704
256        | 0.4853108637 | 0.03156045745 | 0.53     | 5.01613558
512        | 0.6275476315 | 0.05717409239 | 0.63     | 4.852365871
1024       | 0.9063608851 | 0.04540002932 | 0.95     | 2.190429136

### Broadcasting Fields

We don't see a difference between promote and broadcast here, but note in this
case the target fields are low valency. In the case of broadcast, we would have
copying of data for each instance of the target field. This could be quite
expensive, but was not measured in these benchmarks.

Broadcast an int field to depth 2 and project it:

![image](images/broadcast_depth_2.png, "Broadcast a field to depth 2 and project it.")
<!-- source: https://docs.google.com/drawings/d/1JTBoKG7EKhzCbOK1r-rynug78zvVlRZLX5BXhSoIGHs/edit -->

Batch Size | Wall Time    | Wall Time     | CPU Time | CPU Time
---------- | ------------ | ------------- | -------- | -----------
           | avg (ms)     | (stdev)       | avg (ms) | (stdev)
1          | 0.3518526028 | 0.2037769998  | 0.35     | 4.793724854
2          | 0.3390258034 | 0.05497714734 | 0.33     | 4.725815626
4          | 0.3404623303 | 0.07700435376 | 0.34     | 4.760952286
8          | 0.3428831897 | 0.1336404763  | 0.37     | 4.852365871
16         | 0.3434396004 | 0.03195392133 | 0.36     | 4.824181513
32         | 0.3483859585 | 0.05987772519 | 0.35     | 4.793724854
64         | 0.3572916701 | 0.04022134206 | 0.35     | 4.793724854
128        | 0.3759146878 | 0.0233467898  | 0.38     | 4.878317312
256        | 0.4126433665 | 0.0257277493  | 0.4      | 4.923659639
512        | 0.4845570032 | 0.04153148123 | 0.48     | 5.021167316
1024       | 0.6166896177 | 0.03988771645 | 0.62     | 4.878317312

Broadcast an int field to depth 5 and project it:

![image](images/broadcast_depth_5.png, "Broadcast a field to depth 5 and project it.")
<!-- source: https://docs.google.com/drawings/d/1eV_ECW2hdER_9nbdORc4aMld_HHNwXs_rjKjE1oi0E0/edit -->

Batch Size | Wall Time    | Wall Time     | CPU Time | CPU Time
---------- | ------------ | ------------- | -------- | -----------
           | avg (ms)     | (stdev)       | avg (ms) | (stdev)
1          | 0.3408611538 | 0.1291159851  | 0.41     | 4.943110704
2          | 0.337811471  | 0.02195394713 | 0.34     | 4.760952286
4          | 0.3397090239 | 0.02617322913 | 0.35     | 4.793724854
8          | 0.3461196893 | 0.09553572369 | 0.36     | 4.824181513
16         | 0.3444946506 | 0.0996381459  | 0.34     | 4.760952286
32         | 0.3465034659 | 0.07877796899 | 0.36     | 4.824181513
64         | 0.3775842916 | 0.04455864285 | 0.36     | 4.824181513
128        | 0.4091575793 | 0.02269548984 | 0.39     | 4.9020713
256        | 0.4782103971 | 0.03671152727 | 0.49     | 5.024183938
512        | 0.6110463259 | 0.06786895467 | 0.63     | 4.852365871
1024       | 0.8664847137 | 0.0495092042  | 0.91     | 3.785938897

### Rerooting Fields

Rerooting does not incur extra computation cost, because it simply drops the
structure info outside the new root. Thus the results look very similar to
projecting.

Reroot a parent field at depth 1 and project a direct child:

![image](images/reroot_depth_1.png, "Reroot the tree to a field at depth 1 and project a direct field.")
<!-- source: https://docs.google.com/drawings/d/1zTlOFC4IpPPkpdgGdDUJtTxka6dFthR0c2L4wxX-yyA/edit -->

Batch Size | Wall Time    | Wall Time     | CPU Time | CPU Time
---------- | ------------ | ------------- | -------- | -----------
           | avg (ms)     | (stdev)       | avg (ms) | (stdev)
1          | 0.3404013091 | 0.1135904393  | 0.35     | 4.793724854
2          | 0.3387726699 | 0.03198595332 | 0.35     | 4.793724854
4          | 0.3393262327 | 0.0264454256  | 0.36     | 4.824181513
8          | 0.3410624135 | 0.02788640121 | 0.34     | 4.760952286
16         | 0.3447244032 | 0.03587732316 | 0.36     | 4.824181513
32         | 0.3503986569 | 0.09585448124 | 0.37     | 4.852365871
64         | 0.357805646  | 0.02321907052 | 0.34     | 4.760952286
128        | 0.3818715136 | 0.0327633668  | 0.39     | 4.9020713
256        | 0.4239755743 | 0.02716477402 | 0.44     | 4.988876516
512        | 0.5083448561 | 0.08862172331 | 0.55     | 5
1024       | 0.6640480016 | 0.03963619022 | 0.67     | 4.725815626

Reroot a parent field at depth 4 and project a direct child:

![image](images/reroot_depth_4.png, "Reroot the tree to a field at depth 4 and project a direct field.")
<!-- source: https://docs.google.com/drawings/d/1zmUPBimDdP74zhb46P3RqV33K22McdGLouHEbC1lWCY/edit -->

Batch Size | Wall Time    | Wall Time     | CPU Time | CPU Time
---------- | ------------ | ------------- | -------- | -----------
           | avg (ms)     | (stdev)       | avg (ms) | (stdev)
1          | 0.3386712591 | 0.1110942369  | 0.35     | 4.793724854
2          | 0.3376768269 | 0.03301939684 | 0.34     | 4.760952286
4          | 0.3393654749 | 0.02863048876 | 0.35     | 4.793724854
8          | 0.344054353  | 0.09013014869 | 0.34     | 4.760952286
16         | 0.3504192429 | 0.04881439779 | 0.37     | 4.852365871
32         | 0.3620515945 | 0.1283655094  | 0.34     | 4.760952286
64         | 0.3768247641 | 0.04040641729 | 0.39     | 4.9020713
128        | 0.4153456134 | 0.04953889038 | 0.41     | 4.943110704
256        | 0.4884493076 | 0.03172778705 | 0.5      | 5.025189076
512        | 0.6308966316 | 0.04368360826 | 0.75     | 5
1024       | 0.9136783816 | 0.06369979304 | 0.91     | 3.208259543

## Proto to Tensor via. Prensor

These benchmarks uses struct2tensor to decode protos. It turns the prensor
object into Tensors. We look at a different combination of features and
valencies. All features used here are int type. Note that in struct2tensor one
TensorFlow op is created to parse each submessage. And there's no concurrency
inside individual ops. The "flat proto" used below is comparable to
`tf.Examples`, except that it does not have a map inside (each field is a
"feature").

### Flat proto to Dense Tensor

Struct2tensor does not offer a way to directly produce dense tensors from a
prensor tree. In this case, we get ragged tensors, and convert them to dense
tensors. This is why the results show a highly parallelized computation, and are
strictly worse than the ragged tensor case.

1 feature 1 value:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.334     | 0.172     | 0.360    | 4.824
2          | 0.331     | 0.052     | 0.350    | 4.794
4          | 0.332     | 0.043     | 0.350    | 4.794
8          | 0.339     | 0.174     | 0.350    | 4.794
16         | 0.344     | 0.173     | 0.350    | 4.794
32         | 0.344     | 0.035     | 0.370    | 4.852
64         | 0.354     | 0.033     | 0.370    | 4.852
128        | 0.381     | 0.164     | 0.420    | 4.960
256        | 0.418     | 0.130     | 0.440    | 4.989
512        | 0.491     | 0.040     | 0.430    | 4.976
1024       | 0.640     | 0.038     | 0.660    | 4.761

1 feature 100 values each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.338     | 0.169     | 0.340    | 4.761
2          | 0.337     | 0.068     | 0.350    | 4.794
4          | 0.345     | 0.093     | 0.360    | 4.824
8          | 0.358     | 0.034     | 0.380    | 4.878
16         | 0.378     | 0.027     | 0.430    | 4.976
32         | 0.428     | 0.121     | 0.460    | 5.009
64         | 0.557     | 0.074     | 0.620    | 4.878
128        | 0.719     | 0.082     | 0.780    | 4.163
256        | 1.226     | 0.110     | 1.360    | 5.226
512        | 1.973     | 0.126     | 2.330    | 4.726
1024       | 3.779     | 0.243     | 4.730    | 6.172

100 features 1 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 1.954     | 7.248     | 3.330    | 46.646
2          | 1.774     | 0.111     | 1.840    | 5.265
4          | 1.803     | 0.094     | 1.860    | 3.766
8          | 1.824     | 0.099     | 1.900    | 3.892
16         | 1.874     | 0.108     | 1.930    | 3.828
32         | 1.960     | 0.112     | 2.070    | 3.555
64         | 2.114     | 0.225     | 2.200    | 4.264
128        | 2.426     | 0.369     | 2.490    | 5.024
256        | 3.310     | 0.824     | 4.020    | 5.501
512        | 4.674     | 0.762     | 6.000    | 6.195
1024       | 7.882     | 2.298     | 9.970    | 8.463

100 features 100 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 2.312     | 6.974     | 3.830    | 48.598
2          | 2.311     | 0.451     | 2.500    | 6.113
4          | 2.922     | 0.561     | 3.640    | 5.777
8          | 3.775     | 0.938     | 5.250    | 6.571
16         | 5.654     | 1.692     | 7.970    | 8.343
32         | 9.732     | 1.669     | 17.820   | 23.969
64         | 17.679    | 0.942     | 33.870   | 11.070
128        | 32.935    | 1.648     | 63.690   | 15.680
256        | 63.577    | 2.061     | 128.010  | 21.106
512        | 123.537   | 4.260     | 275.920  | 34.221
1024       | 254.632   | 13.373    | 771.720  | 56.337

### Flat proto to Sparse Tensor

We see some parallelization in the sparse case. This is likely due to the way
coo indexes are created.

1 feature 1 value:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.314     | 0.098     | 0.320    | 4.688
2          | 0.313     | 0.057     | 0.330    | 4.726
4          | 0.314     | 0.022     | 0.320    | 4.688
8          | 0.317     | 0.099     | 0.330    | 4.726
16         | 0.318     | 0.067     | 0.330    | 4.726
32         | 0.323     | 0.103     | 0.310    | 4.648
64         | 0.336     | 0.138     | 0.340    | 4.761
128        | 0.352     | 0.026     | 0.340    | 4.761
256        | 0.389     | 0.098     | 0.400    | 4.924
512        | 0.457     | 0.085     | 0.460    | 5.009
1024       | 0.600     | 0.167     | 0.610    | 4.902

1 feature 100 values each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.325     | 0.167     | 0.320    | 4.688
2          | 0.318     | 0.024     | 0.360    | 4.824
4          | 0.324     | 0.020     | 0.330    | 4.726
8          | 0.341     | 0.046     | 0.350    | 4.794
16         | 0.365     | 0.021     | 0.360    | 4.824
32         | 0.416     | 0.024     | 0.480    | 5.218
64         | 0.547     | 0.038     | 0.550    | 5.000
128        | 0.723     | 0.037     | 0.740    | 4.408
256        | 1.053     | 0.088     | 1.480    | 5.218
512        | 1.694     | 0.152     | 2.770    | 5.096
1024       | 3.327     | 0.398     | 5.280    | 5.140

100 features 1 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.444     | 0.298     | 0.460    | 5.207
2          | 0.437     | 0.028     | 0.410    | 4.943
4          | 0.448     | 0.040     | 0.430    | 4.976
8          | 0.465     | 0.034     | 0.460    | 5.009
16         | 0.506     | 0.048     | 0.510    | 5.024
32         | 0.585     | 0.036     | 0.570    | 4.976
64         | 0.742     | 0.043     | 0.770    | 4.230
128        | 1.081     | 0.076     | 1.080    | 3.674
256        | 1.938     | 0.645     | 2.090    | 4.943
512        | 3.375     | 0.502     | 4.410    | 5.522
1024       | 6.011     | 1.052     | 8.510    | 6.435

100 features 100 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.691     | 0.157     | 0.760    | 5.148
2          | 0.954     | 0.047     | 1.040    | 4.000
4          | 1.396     | 0.629     | 2.010    | 4.819
8          | 2.231     | 0.265     | 3.550    | 6.093
16         | 4.024     | 0.881     | 7.180    | 6.417
32         | 7.028     | 1.654     | 13.820   | 8.573
64         | 15.389    | 0.962     | 29.600   | 11.721
128        | 30.858    | 1.587     | 58.850   | 10.286
256        | 62.040    | 2.098     | 128.500  | 22.134
512        | 123.279   | 4.244     | 270.940  | 33.419
1024       | 243.880   | 10.169    | 564.540  | 259.391

### Flat proto to Ragged Tensor

In some sense, ragged is the most natural way to represent a prensor tree (and
thus should be the most efficient). This is because prensor trees have the same
internal representation as ragged tensors. We see that there is little
parallelization going on here. It has the lowest (fastest) wall and cpu time
here, compared to sparse and dense tensors.

1 feature 1 value:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.328     | 0.195     | 0.340    | 4.761
2          | 0.313     | 0.087     | 0.310    | 4.648
4          | 0.327     | 0.112     | 0.320    | 4.688
8          | 0.328     | 0.098     | 0.350    | 4.794
16         | 0.326     | 0.205     | 0.320    | 4.688
32         | 0.326     | 0.169     | 0.340    | 4.761
64         | 0.329     | 0.026     | 0.330    | 4.726
128        | 0.350     | 0.027     | 0.360    | 4.824
256        | 0.388     | 0.030     | 0.380    | 4.878
512        | 0.460     | 0.044     | 0.480    | 5.021
1024       | 0.599     | 0.049     | 0.600    | 4.924

1 feature 100 values each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.313     | 0.108     | 0.320    | 4.688
2          | 0.313     | 0.020     | 0.320    | 4.688
4          | 0.319     | 0.023     | 0.330    | 4.726
8          | 0.330     | 0.026     | 0.350    | 4.794
16         | 0.345     | 0.026     | 0.380    | 4.878
32         | 0.379     | 0.025     | 0.390    | 4.902
64         | 0.473     | 0.040     | 0.510    | 5.024
128        | 0.580     | 0.036     | 0.580    | 4.960
256        | 0.926     | 0.084     | 0.950    | 2.190
512        | 1.443     | 0.311     | 1.470    | 5.016
1024       | 2.724     | 0.302     | 2.730    | 5.096

100 features 1 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.350     | 0.141     | 0.340    | 4.761
2          | 0.353     | 0.028     | 0.370    | 4.852
4          | 0.352     | 0.096     | 0.340    | 4.761
8          | 0.375     | 0.034     | 0.410    | 4.943
16         | 0.409     | 0.046     | 0.530    | 5.016
32         | 0.472     | 0.032     | 0.480    | 5.021
64         | 0.602     | 0.094     | 0.620    | 4.878
128        | 0.867     | 0.104     | 0.930    | 3.828
256        | 1.551     | 0.217     | 1.550    | 5.198
512        | 2.763     | 0.180     | 2.800    | 4.924
1024       | 5.145     | 0.709     | 5.270    | 5.478

100 features 100 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.518     | 0.140     | 0.480    | 5.021
2          | 0.691     | 0.062     | 0.720    | 4.513
4          | 1.037     | 0.059     | 1.060    | 3.712
8          | 1.747     | 0.131     | 1.770    | 4.894
16         | 3.086     | 0.190     | 3.130    | 4.416
32         | 5.939     | 0.677     | 6.120    | 4.557
64         | 13.216    | 1.378     | 13.640   | 6.439
128        | 28.063    | 1.545     | 29.130   | 13.533
256        | 56.429    | 2.679     | 58.600   | 19.437
512        | 114.878   | 9.048     | 119.320  | 32.283
1024       | 225.467   | 9.514     | 233.680  | 33.086

### Deep proto to Dense Tensor

The results here are slightly slower than projecting alone. This confirms that
we are spending a lot of time on parsing. The delta between these benchmarks and
the project deep benchmarks illustrates the cost in constructing dense tensors.

1 feature 1 value with depth 1:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.334     | 0.148     | 0.340    | 4.761
2          | 0.333     | 0.041     | 0.340    | 4.761
4          | 0.333     | 0.030     | 0.350    | 4.794
8          | 0.336     | 0.103     | 0.350    | 4.794
16         | 0.345     | 0.103     | 0.360    | 4.824
32         | 0.346     | 0.146     | 0.360    | 4.824
64         | 0.351     | 0.075     | 0.340    | 4.761
128        | 0.369     | 0.115     | 0.390    | 4.902
256        | 0.401     | 0.035     | 0.420    | 4.960
512        | 0.458     | 0.035     | 0.480    | 5.021
1024       | 0.566     | 0.038     | 0.590    | 4.943

1 feature 1 value with depth 2:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.354     | 0.258     | 0.400    | 4.924
2          | 0.349     | 0.088     | 0.320    | 4.688
4          | 0.348     | 0.030     | 0.350    | 4.794
8          | 0.349     | 0.035     | 0.350    | 4.794
16         | 0.354     | 0.027     | 0.390    | 4.902
32         | 0.361     | 0.067     | 0.380    | 4.878
64         | 0.378     | 0.049     | 0.430    | 4.976
128        | 0.399     | 0.050     | 0.470    | 5.016
256        | 0.446     | 0.042     | 0.490    | 5.024
512        | 0.533     | 0.051     | 0.590    | 4.943
1024       | 0.689     | 0.129     | 0.750    | 4.578

1 feature 1 value with depth 3:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.366     | 0.339     | 0.370    | 4.852
2          | 0.366     | 0.079     | 0.360    | 4.824
4          | 0.367     | 0.028     | 0.380    | 4.878
8          | 0.371     | 0.056     | 0.350    | 4.794
16         | 0.374     | 0.028     | 0.380    | 4.878
32         | 0.393     | 0.112     | 0.490    | 5.024
64         | 0.395     | 0.072     | 0.450    | 5.000
128        | 0.439     | 0.114     | 0.540    | 5.009
256        | 0.507     | 0.147     | 0.600    | 4.924
512        | 0.629     | 0.036     | 0.700    | 4.606
1024       | 0.856     | 0.060     | 0.970    | 4.370

1 feature 1 value with depth 4:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.389     | 0.347     | 0.420    | 5.352
2          | 0.378     | 0.033     | 0.390    | 4.902
4          | 0.382     | 0.034     | 0.380    | 4.878
8          | 0.384     | 0.029     | 0.460    | 5.009
16         | 0.391     | 0.067     | 0.470    | 5.016
32         | 0.415     | 0.144     | 0.550    | 5.000
64         | 0.423     | 0.086     | 0.540    | 5.009
128        | 0.467     | 0.102     | 0.560    | 4.989
256        | 0.552     | 0.113     | 0.680    | 4.688
512        | 0.688     | 0.060     | 0.820    | 3.861
1024       | 0.927     | 0.059     | 1.060    | 4.221

1 feature 1 value with depth 5:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.405     | 0.401     | 0.450    | 5.752
2          | 0.395     | 0.110     | 0.390    | 4.902
4          | 0.400     | 0.033     | 0.420    | 4.960
8          | 0.401     | 0.033     | 0.420    | 4.960
16         | 0.408     | 0.028     | 0.460    | 5.009
32         | 0.425     | 0.125     | 0.530    | 5.016
64         | 0.430     | 0.073     | 0.600    | 4.924
128        | 0.476     | 0.038     | 0.640    | 4.824
256        | 0.577     | 0.116     | 0.750    | 4.578
512        | 0.750     | 0.060     | 0.900    | 4.144
1024       | 1.068     | 0.053     | 1.200    | 4.714

### Deep proto to Sparse Tensor

The results here are slightly slower than projecting alone. This confirms that
we are spending a lot of time on parsing. The delta between these benchmarks and
the project deep benchmarks illustrates the cost in constructing sparse tensors.

1 feature 1 value with depth 1:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.313     | 0.076     | 0.320    | 4.688
2          | 0.312     | 0.024     | 0.310    | 4.648
4          | 0.313     | 0.033     | 0.320    | 4.688
8          | 0.321     | 0.120     | 0.320    | 4.688
16         | 0.323     | 0.165     | 0.320    | 4.688
32         | 0.322     | 0.068     | 0.320    | 4.688
64         | 0.329     | 0.029     | 0.340    | 4.761
128        | 0.346     | 0.057     | 0.350    | 4.794
256        | 0.379     | 0.069     | 0.390    | 4.902
512        | 0.436     | 0.038     | 0.430    | 4.976
1024       | 0.548     | 0.040     | 0.520    | 5.021

1 feature 1 value with depth 2:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.320     | 0.120     | 0.330    | 4.726
2          | 0.316     | 0.024     | 0.320    | 4.688
4          | 0.319     | 0.077     | 0.330    | 4.726
8          | 0.319     | 0.043     | 0.320    | 4.688
16         | 0.322     | 0.055     | 0.320    | 4.688
32         | 0.331     | 0.050     | 0.360    | 4.824
64         | 0.344     | 0.070     | 0.380    | 4.878
128        | 0.373     | 0.086     | 0.400    | 4.924
256        | 0.414     | 0.080     | 0.440    | 4.989
512        | 0.508     | 0.075     | 0.560    | 4.989
1024       | 0.673     | 0.063     | 0.730    | 4.462

1 feature 1 value with depth 3:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.324     | 0.203     | 0.330    | 4.726
2          | 0.317     | 0.038     | 0.330    | 4.726
4          | 0.318     | 0.025     | 0.330    | 4.726
8          | 0.320     | 0.027     | 0.310    | 4.648
16         | 0.323     | 0.034     | 0.300    | 4.606
32         | 0.336     | 0.063     | 0.350    | 4.794
64         | 0.359     | 0.117     | 0.360    | 4.824
128        | 0.392     | 0.062     | 0.430    | 4.976
256        | 0.461     | 0.040     | 0.500    | 5.025
512        | 0.593     | 0.075     | 0.630    | 4.852
1024       | 0.815     | 0.060     | 0.850    | 3.589

1 feature 1 value with depth 4:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.323     | 0.195     | 0.340    | 4.761
2          | 0.319     | 0.044     | 0.340    | 4.761
4          | 0.322     | 0.077     | 0.370    | 5.056
8          | 0.326     | 0.083     | 0.370    | 4.852
16         | 0.328     | 0.062     | 0.340    | 4.761
32         | 0.346     | 0.102     | 0.350    | 4.794
64         | 0.368     | 0.045     | 0.410    | 4.943
128        | 0.415     | 0.060     | 0.480    | 5.021
256        | 0.504     | 0.071     | 0.530    | 5.404
512        | 0.663     | 0.072     | 0.760    | 4.292
1024       | 0.939     | 0.073     | 1.090    | 5.143

1 feature 1 value with depth 5:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.323     | 0.191     | 0.330    | 4.726
2          | 0.321     | 0.085     | 0.310    | 4.648
4          | 0.324     | 0.109     | 0.330    | 4.726
8          | 0.323     | 0.039     | 0.330    | 4.726
16         | 0.329     | 0.053     | 0.320    | 4.688
32         | 0.354     | 0.201     | 0.360    | 4.824
64         | 0.384     | 0.175     | 0.440    | 4.989
128        | 0.425     | 0.068     | 0.500    | 5.025
256        | 0.524     | 0.075     | 0.570    | 4.976
512        | 0.696     | 0.080     | 0.800    | 4.020
1024       | 0.998     | 0.070     | 1.130    | 3.933

### Deep proto to Ragged Tensor

The results here are slightly slower than projecting alone. This confirms that
we are spending a lot of time on parsing. Interestingly, the results here are
faster than projectioni alone. This is likely a combination of variance, and how
the project benchmark is tested. In order for TensorFlow to trace the graph, and
get a benchmark of the prensor operations, we need to use control dependencies.
We have a control dependency that constructs a `tf.constant(1)`. Likely
constructing that constant has similar cost to constructing the ragged tensor,
since the components of the ragged tensor are already created.

1 feature 1 value with depth 1:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.320     | 0.120     | 0.320    | 4.688
2          | 0.323     | 0.078     | 0.320    | 4.688
4          | 0.320     | 0.078     | 0.310    | 4.648
8          | 0.326     | 0.113     | 0.320    | 4.688
16         | 0.331     | 0.125     | 0.380    | 5.081
32         | 0.317     | 0.180     | 0.320    | 4.688
64         | 0.322     | 0.051     | 0.330    | 4.726
128        | 0.338     | 0.065     | 0.340    | 4.761
256        | 0.372     | 0.077     | 0.370    | 4.852
512        | 0.427     | 0.028     | 0.420    | 4.960
1024       | 0.532     | 0.039     | 0.530    | 5.016

1 feature 1 value with depth 2:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.326     | 0.135     | 0.330    | 4.726
2          | 0.332     | 0.122     | 0.340    | 4.761
4          | 0.330     | 0.127     | 0.330    | 4.726
8          | 0.323     | 0.184     | 0.310    | 4.648
16         | 0.316     | 0.089     | 0.330    | 4.726
32         | 0.318     | 0.025     | 0.320    | 4.688
64         | 0.329     | 0.032     | 0.340    | 4.761
128        | 0.352     | 0.037     | 0.350    | 4.794
256        | 0.380     | 0.074     | 0.400    | 4.924
512        | 0.464     | 0.119     | 0.480    | 5.021
1024       | 0.617     | 0.177     | 0.620    | 4.878

1 feature 1 value with depth 3:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.307     | 0.154     | 0.300    | 4.606
2          | 0.311     | 0.034     | 0.320    | 4.688
4          | 0.311     | 0.023     | 0.310    | 4.648
8          | 0.313     | 0.024     | 0.310    | 4.648
16         | 0.317     | 0.024     | 0.330    | 4.726
32         | 0.324     | 0.024     | 0.300    | 4.606
64         | 0.337     | 0.062     | 0.340    | 4.761
128        | 0.361     | 0.146     | 0.360    | 4.824
256        | 0.424     | 0.026     | 0.420    | 4.960
512        | 0.531     | 0.031     | 0.530    | 5.016
1024       | 0.733     | 0.040     | 0.760    | 4.292

1 feature 1 value with depth 4:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.316     | 0.175     | 0.320    | 4.688
2          | 0.310     | 0.025     | 0.310    | 4.648
4          | 0.313     | 0.032     | 0.320    | 4.688
8          | 0.314     | 0.021     | 0.320    | 4.688
16         | 0.321     | 0.119     | 0.330    | 4.726
32         | 0.328     | 0.025     | 0.350    | 4.794
64         | 0.343     | 0.024     | 0.380    | 4.878
128        | 0.379     | 0.026     | 0.390    | 4.902
256        | 0.445     | 0.027     | 0.440    | 4.989
512        | 0.574     | 0.034     | 0.590    | 4.943
1024       | 0.825     | 0.056     | 0.860    | 3.766

1 feature 1 value with depth 5:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.312     | 0.129     | 0.310    | 4.648
2          | 0.310     | 0.022     | 0.310    | 4.648
4          | 0.315     | 0.108     | 0.310    | 4.648
8          | 0.316     | 0.027     | 0.330    | 4.726
16         | 0.320     | 0.019     | 0.330    | 4.726
32         | 0.322     | 0.128     | 0.340    | 4.761
64         | 0.337     | 0.085     | 0.350    | 4.794
128        | 0.371     | 0.056     | 0.390    | 4.902
256        | 0.444     | 0.079     | 0.470    | 5.016
512        | 0.579     | 0.072     | 0.620    | 4.878
1024       | 0.883     | 0.174     | 0.870    | 3.933

## tf.Example to Tensor

These benchmarks use `tf.io.parse_example` to decode tf.Examples into Tensors.
`tf.io.parse_example`'s op is highly parallelized, which is reflected in the
results.

### FixedLenFeature

We see much much faster wall times compared to prensor's to dense tensor. The
CPU time is faster too, however keep in mind that prensors had to go through
that extra ragged -> dense conversion step.

1 feature 1 value:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.317     | 0.123     | 0.340    | 4.761
2          | 0.331     | 0.056     | 0.350    | 4.794
4          | 0.336     | 0.108     | 0.360    | 4.824
8          | 0.340     | 0.043     | 0.380    | 4.878
16         | 0.341     | 0.027     | 0.370    | 4.852
32         | 0.347     | 0.032     | 0.420    | 4.960
64         | 0.359     | 0.071     | 0.430    | 4.976
128        | 0.380     | 0.040     | 0.530    | 5.016
256        | 0.416     | 0.080     | 0.670    | 4.726
512        | 0.479     | 0.151     | 0.930    | 4.768
1024       | 0.656     | 0.039     | 1.550    | 5.000

1 feature 100 values each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.299     | 0.065     | 0.320    | 4.688
2          | 0.325     | 0.098     | 0.340    | 4.761
4          | 0.333     | 0.051     | 0.370    | 4.852
8          | 0.341     | 0.062     | 0.370    | 4.852
16         | 0.345     | 0.045     | 0.420    | 4.960
32         | 0.350     | 0.027     | 0.410    | 4.943
64         | 0.361     | 0.029     | 0.480    | 5.021
128        | 0.386     | 0.034     | 0.610    | 4.902
256        | 0.425     | 0.088     | 0.750    | 4.794
512        | 0.504     | 0.074     | 1.090    | 6.528
1024       | 0.670     | 0.083     | 2.200    | 6.963

100 features 1 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.355     | 0.133     | 0.370    | 4.852
2          | 0.377     | 0.110     | 0.380    | 4.878
4          | 0.390     | 0.063     | 0.450    | 5.000
8          | 0.405     | 0.047     | 0.540    | 5.009
16         | 0.417     | 0.058     | 0.650    | 4.794
32         | 0.446     | 0.066     | 0.800    | 4.020
64         | 0.501     | 0.022     | 1.210    | 6.860
128        | 0.618     | 0.021     | 2.010    | 3.332
256        | 0.867     | 0.073     | 5.340    | 4.761
512        | 1.359     | 0.085     | 10.760   | 5.148
1024       | 2.377     | 0.316     | 20.970   | 6.269

100 features 100 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.466     | 0.136     | 0.470    | 5.016
2          | 0.491     | 0.107     | 0.590    | 4.943
4          | 0.520     | 0.108     | 0.910    | 4.286
8          | 0.549     | 0.143     | 1.480    | 5.218
16         | 0.801     | 0.103     | 2.800    | 4.924
32         | 1.069     | 0.102     | 6.210    | 6.711
64         | 1.730     | 0.184     | 12.670   | 5.515
128        | 3.695     | 0.456     | 28.260   | 7.333
256        | 6.991     | 0.703     | 57.400   | 8.040
512        | 12.863    | 1.347     | 112.190  | 9.287
1024       | 24.120    | 2.290     | 217.860  | 16.516

### VarLenFeature

The wall time and CPU time are both faster than prensor's to sparse. This is
also slightly unfair to prensors because of the parent indicies -> coo
conversion.

1 feature 1 value:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.313     | 0.053     | 0.300    | 4.606
2          | 0.325     | 0.071     | 0.350    | 4.794
4          | 0.337     | 0.126     | 0.350    | 4.794
8          | 0.340     | 0.037     | 0.400    | 4.924
16         | 0.343     | 0.032     | 0.390    | 4.902
32         | 0.349     | 0.035     | 0.410    | 4.943
64         | 0.359     | 0.025     | 0.450    | 5.000
128        | 0.384     | 0.081     | 0.550    | 5.000
256        | 0.425     | 0.016     | 0.690    | 4.861
512        | 0.482     | 0.148     | 0.890    | 4.239
1024       | 0.663     | 0.037     | 1.600    | 4.924

1 feature 100 values each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.317     | 0.163     | 0.340    | 4.761
2          | 0.326     | 0.156     | 0.350    | 4.794
4          | 0.334     | 0.065     | 0.380    | 4.878
8          | 0.342     | 0.041     | 0.390    | 4.902
16         | 0.351     | 0.061     | 0.390    | 4.902
32         | 0.358     | 0.076     | 0.410    | 4.943
64         | 0.379     | 0.036     | 0.530    | 5.016
128        | 0.413     | 0.041     | 0.600    | 4.924
256        | 0.477     | 0.115     | 0.860    | 4.269
512        | 0.585     | 0.020     | 1.210    | 4.094
1024       | 0.852     | 0.077     | 2.590    | 6.371

100 features 1 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.377     | 0.186     | 0.370    | 4.852
2          | 0.396     | 0.036     | 0.420    | 4.960
4          | 0.425     | 0.061     | 0.500    | 5.025
8          | 0.464     | 0.032     | 0.660    | 4.761
16         | 0.507     | 0.076     | 0.840    | 6.470
32         | 0.548     | 0.184     | 1.030    | 5.214
64         | 0.644     | 0.094     | 1.330    | 4.726
128        | 0.812     | 0.401     | 2.220    | 4.399
256        | 1.211     | 0.116     | 6.160    | 4.866
512        | 1.952     | 0.174     | 12.490   | 7.587
1024       | 3.732     | 0.335     | 24.460   | 6.578

100 features 100 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.522     | 0.112     | 0.500    | 5.025
2          | 0.565     | 0.078     | 0.740    | 4.408
4          | 0.646     | 0.213     | 1.100    | 3.892
8          | 0.765     | 0.174     | 1.870    | 5.801
16         | 1.238     | 0.158     | 3.540    | 5.581
32         | 2.320     | 0.391     | 9.140    | 5.689
64         | 5.023     | 0.354     | 19.600   | 7.785
128        | 9.410     | 0.517     | 40.320   | 9.198
256        | 17.257    | 1.050     | 80.740   | 11.859
512        | 32.434    | 1.139     | 166.210  | 14.927
1024       | 60.879    | 3.887     | 329.010  | 21.719

### RaggedFeature

This is probably the most fair comparision for `tf.io.parse_example` vs
prensors. The wall time is still much faster for `tf.io.parse_example`, however
we see prensors has an edge in terms of cpu time. This is especially true for
high number of features. This is because in a `tf.Example` we need a proto map
for each feature. Whereas the data in it's natural proto would not require a map
field. Parsing a proto map is expensive.

These benchmarks show that prensors has the potential to outclass `tf.Examples`.
Given `tf.Examples` many years of headstart in adoption and optimization, I'm
excited to see where prensors can grow to outperform it.

1 feature 1 value:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.312     | 0.039     | 0.320    | 4.688
2          | 0.327     | 0.095     | 0.360    | 4.824
4          | 0.334     | 0.027     | 0.370    | 4.852
8          | 0.338     | 0.036     | 0.400    | 4.924
16         | 0.343     | 0.031     | 0.380    | 4.878
32         | 0.351     | 0.033     | 0.410    | 4.943
64         | 0.358     | 0.030     | 0.430    | 4.976
128        | 0.379     | 0.028     | 0.550    | 5.000
256        | 0.415     | 0.062     | 0.680    | 4.688
512        | 0.475     | 0.028     | 0.900    | 3.624
1024       | 0.661     | 0.035     | 1.560    | 4.989

1 feature 100 values each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.318     | 0.170     | 0.320    | 4.688
2          | 0.327     | 0.088     | 0.350    | 4.794
4          | 0.336     | 0.032     | 0.350    | 4.794
8          | 0.339     | 0.039     | 0.380    | 4.878
16         | 0.347     | 0.036     | 0.400    | 4.924
32         | 0.354     | 0.030     | 0.430    | 4.976
64         | 0.370     | 0.051     | 0.470    | 5.016
128        | 0.402     | 0.054     | 0.640    | 4.824
256        | 0.446     | 0.080     | 0.760    | 4.292
512        | 0.542     | 0.031     | 1.170    | 5.329
1024       | 0.768     | 0.099     | 2.420    | 5.352

100 features 1 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.365     | 0.144     | 0.350    | 4.794
2          | 0.387     | 0.072     | 0.410    | 4.943
4          | 0.408     | 0.095     | 0.480    | 5.021
8          | 0.435     | 0.063     | 0.580    | 4.960
16         | 0.452     | 0.074     | 0.690    | 4.648
32         | 0.489     | 0.113     | 0.960    | 4.477
64         | 0.572     | 0.081     | 1.330    | 4.935
128        | 0.699     | 0.186     | 2.090    | 4.516
256        | 1.040     | 0.085     | 6.060    | 8.741
512        | 1.686     | 0.205     | 12.000   | 6.030
1024       | 3.157     | 0.394     | 23.710   | 7.148

100 features 100 value each:

Batch Size | Wall Time | Wall Time | CPU Time | CPU Time
---------- | --------- | --------- | -------- | --------
           | avg (ms)  | (stdev)   | avg (ms) | (stdev)
1          | 0.493     | 0.120     | 0.430    | 4.976
2          | 0.536     | 0.125     | 0.680    | 4.899
4          | 0.584     | 0.152     | 1.010    | 3.332
8          | 0.712     | 0.350     | 1.810    | 4.861
16         | 1.013     | 0.139     | 3.350    | 18.056
32         | 1.537     | 0.226     | 7.860    | 6.966
64         | 3.319     | 0.452     | 17.390   | 6.651
128        | 6.796     | 0.563     | 37.480   | 9.690
256        | 12.547    | 0.583     | 75.590   | 11.110
512        | 23.602    | 0.843     | 156.620  | 12.373
1024       | 43.847    | 2.155     | 309.940  | 22.955

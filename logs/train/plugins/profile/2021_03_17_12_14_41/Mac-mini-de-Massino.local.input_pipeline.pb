	/?$?ߚ@/?$?ߚ@!/?$?ߚ@	?f???t???f???t??!?f???t??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$/?$?ߚ@?n?????Ah??|???@YD?l???@*	     ??@2|
EIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::MapJ+??@!??????U@)?x?&1?@1?%?g[?U@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::BatchV2L7?A`e@!?U(??X@)?MbX9??1?Ij&@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip??ʡ?@!Ѐ<b&V@)????????1?H{?T??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle??|?5?@!??lƪ!V@)?~j?t???1?n$0d???:Preprocessing2?
MIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[1]::TensorSlice/?$???!??????)/?$???1??????:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::Map::TensorSlicey?&1???!]?????)y?&1???1]?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??? ?r@!1IE?~?X@)9??v????1{"???$??:Preprocessing2F
Iterator::Model{?G?z@!      Y@)????Mb??1!>?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?f???t??I?u-??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?n??????n?????!?n?????      ??!       "      ??!       *      ??!       2	h??|???@h??|???@!h??|???@:      ??!       B      ??!       J	D?l???@D?l???@!D?l???@R      ??!       Z	D?l???@D?l???@!D?l???@b      ??!       JCPU_ONLYY?f???t??b q?u-??X@
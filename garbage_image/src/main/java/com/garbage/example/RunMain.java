package com.garbage.example;
import com.alibaba.tianchi.garbage_image_util.DebugFlatMap;
import com.alibaba.tianchi.garbage_image_util.ImageClassSink;
import com.alibaba.tianchi.garbage_image_util.ImageDirSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class RunMain {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        flinkEnv.setParallelism(1);
        ImageDirSource source = new ImageDirSource();
        ExtendedInferenceModel model = new ExtendedInferenceModel();

        String modelPath = System.getenv("IMAGE_MODEL_PATH");
        String modelPathpackage = "/home/parallels/saved_model.tar.gz";

        flinkEnv.addSource(source).setParallelism(1)
                .flatMap(new ProcessingFlatMap()).setParallelism(1)
                .flatMap(new UpdateDebugFlatMap(System.getenv("IMAGE_INPUT_PATH"),System.getenv("IMAGE_MODEL_PACKAGE_PATH")))
                //.flatMap(new UpdateDebugFlatMap(System.getenv("IMAGE_INPUT_PATH"),modelPathpackage))
                .setParallelism(1)
                //.flatMap(new DebugFlatMap()).setParallelism(4)
                .addSink(new ImageClassSink()).setParallelism(1);
        flinkEnv.execute();

    }
}


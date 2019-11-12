package com.garbage.example;

import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.util.Collector;
import scala.Tuple2;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;

import javax.imageio.ImageIO;
import java.awt.image.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.FloatBuffer;

public class ProcessingFlatMap extends RichFlatMapFunction<ImageData, Tuple2<String, JTensor>> {






    public static float[] fromHWC2CHW(float[] data) {
        float[] resArray = new float[3 * 331 * 331];
        for (int h = 0; h < 331; h++) {
            for (int w=0; w < 331; w++) {
                for (int c =0; c < 3; c++) {
                    resArray[c * 331 * 331 + h * 331 + w] = data[h * 331 * 3 + w * 3 + c];
                }
            }
        }
        return resArray;
    }




    public  float[] convert_cindy2(byte[] buffer) throws IOException{
        BufferedImage bufferedImage = ImageIO.read(new ByteArrayInputStream(buffer));
        ColorProcessor colorProcessor = new ColorProcessor(bufferedImage);
        ColorProcessor imageProcessor = (ColorProcessor)colorProcessor.convertToRGB();
        imageProcessor = (ColorProcessor)imageProcessor.resize(331, 331, true);
        int channelNum = imageProcessor.getNChannels();
        FloatBuffer floatBuffer = FloatBuffer.allocate(331*331*channelNum);
        FloatProcessor[] fps = new FloatProcessor[channelNum];
        float[][][] floats = new float[channelNum][331][331];
        for(int i = 0; i < channelNum; i ++){
            fps[i] = imageProcessor.toFloat(i, null);
            floats[i] = fps[i].getFloatArray();
        }
        for (int k = 0; k < 331; k++ ) {
            for(int t = 0; t < 331; t++) {
                for(int j = 0; j < channelNum; j++) {
                    floatBuffer.put(floats[j][t][k]);
                }
            }
        }
        ((Buffer)floatBuffer).flip();
        return fromHWC2CHW(floatBuffer.array()) ;
    }




    @Override
    public void flatMap(ImageData imageData, Collector<Tuple2<String, JTensor>> collector) throws Exception {
        JTensor jTensor = new JTensor();
        int[] shape = {1, 331, 331,3};
        jTensor.setShape(shape);
        jTensor.setData(convert_cindy2(imageData.getImage()));

        collector.collect(new Tuple2<>(imageData.getId(), jTensor));
    }
}
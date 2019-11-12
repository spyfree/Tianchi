package com.garbage.example;

import com.alibaba.tianchi.garbage_image_util.DebugFlatMap;
import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;
import scala.Tuple2;


import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;


public class UpdateDebugFlatMap extends RichFlatMapFunction<Tuple2<String, JTensor>, IdLabel> {
    private ExtendedInferenceModel model;
    private HashMap<Integer, String> hashMap;
    private String modelPath;
    UpdateDebugFlatMap(String imagePath, String modelPath) throws FileNotFoundException {
        this.modelPath = modelPath;
        hashMap = generateDict();
    }
    UpdateDebugFlatMap(){}

    public HashMap<Integer, String> generateDict() throws FileNotFoundException {
        HashMap hashMap = new HashMap();
        InputStream is = this.getClass().getResourceAsStream("/class_index.txt");
        //InputStream is = new FileInputStream("/class_index.txt");
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(is));
            String line = br.readLine();
            while (line != null) {
                String[] arr = line.split(" ");
                hashMap.put(Integer.parseInt(arr[1]), arr[0]);
                line = br.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return hashMap;
    }

    public int getMaxIndex(float[] floats) {
        int maxIndex = -1;
        float maxF = -1;
        for (int i=0; i<floats.length; i++) {
            if (floats[i] > maxF) {
                maxIndex = i;
                maxF = floats[i];
            }
        }
        return maxIndex;
    }

    public void printFloatArr(float[] floats){
        for (float f : floats) {
            System.out.print(String.valueOf(f) + ", ");
        }
        System.out.println();
    }

    public void printLRFloatArr(float[] floats, int left, int right){
        for (int i=left; i < right ; i ++) {
            System.out.print(String.valueOf(floats[i]) + ", ");
        }
        System.out.println();
    }

    @Override
    public void flatMap(Tuple2<String, JTensor> value, Collector<IdLabel> out) throws Exception {
        List<JTensor> data = Arrays.asList(value._2);
        List<List<JTensor>> inputs = new ArrayList<>();
        //System.out.print(data);
        inputs.add(data);
        float[] outputData = model.doPredict(inputs).get(0).get(0).getData();
        printFloatArr(outputData);
        int index = getMaxIndex(outputData);
        System.out.println(index);
        String label = hashMap.get(index);
        IdLabel idLabel = new IdLabel(value._1, label);
        out.collect(idLabel);
    }

    private static class ExtendedInferenceModel extends AbstractInferenceModel {
        ExtendedInferenceModel() {
            super();
        }
    }

    public static byte[] toByteArray(String filepath)throws IOException{
        long filesize = new File(filepath).length();
        byte[] savedModelTarBytes = new byte[(int)filesize];
        InputStream inputStream = new FileInputStream(filepath);
        inputStream.read(savedModelTarBytes);
        return savedModelTarBytes;
    }
    public float[] convertToFloat(byte[] buffer) {
        float[] fArr = new float[buffer.length];  // 4 bytes per float
        for (int i = 0; i < fArr.length; i++)
        {
            fArr[i] = buffer[i];
        }
        return fArr;
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        model = new ExtendedInferenceModel();
        //int[] inputShape = {1,224, 224,3};
        int[] inputShape = {1,331, 331,3};
        //float[] meanValues = {123.68f, 116.78f, 183.94f};
       //float[] meanValues = {123.68f, 116.78f, 103.94f};
        //float[] meanValues = {103.94f, 116.78f, 123.68f};
//        float[] meanValues = {0.5f, 0.4f, 0.67f};
        float[] meanValues = {127.5f, 127.5f, 127.5f};
        try {
            //model.doLoadTF(toByteArray(modelPath), inputShape, true, meanValues, 1.0f, "input_1");
            model.doLoadTF(toByteArray(modelPath), inputShape, false, meanValues, 127.5f, "input_1");
            //model.doLoadTF(toByteArray(modelPath), inputShape, true, meanValues, 1.0f, "DecodeJpeg/contents:0");
            //model.loadTF(toByteArray(modelPath+"saved_model.pb"), "saved_model", toByteArray(modelPath+"variables/checkpoint"), inputShape, true, meanValues, 1);
        } catch (IOException e) {
            e.printStackTrace();
        }
        super.open(parameters);
    }

    @Override
    public void close() throws Exception {
        super.close();
    }
}
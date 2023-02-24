# 模型发布到 ModelScope

1. 在ModelScope官网点击右上角 "+创建" -> "创建模型"
2. 填写模型基本信息，点击"创建模型"
3. 本地运行命令，初始化模型仓库

   ```
   git clone https://www.modelscope.cn/${model_id}.git
   ```

4. 将训练后保存的`output` 文件夹中的所有文件拷贝到模型仓库
5. 编辑README.md，填写模型卡片说明，详见 [如何撰写好用的模型卡片](https://www.modelscope.cn/docs/如何撰写好用的模型卡片)
6. 在模型仓库，运行以下命令，将模型上传到ModelScope

    ```
    git add .
    git commit -m "upload my model"
    git push origin master
    ```

7. 然后就可以直接使用ModelScope进行推理了，详见 [模型推理](./model_inference_zh.md)

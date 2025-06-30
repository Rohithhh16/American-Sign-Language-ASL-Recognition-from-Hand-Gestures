from module import *

pipeline1 = pipeline(imgcls_path="imgcls_weights/quant_modles/efficient_asl_16_2.tflite",
                     source_path=0)
objdetmodel = pipeline1.load_objectdetection_model()
imgclsmodel = pipeline1.load_imageclassification_model()
database = pipeline1.load_database()
pipeline1.run(model1=objdetmodel,model2=imgclsmodel,db=database)





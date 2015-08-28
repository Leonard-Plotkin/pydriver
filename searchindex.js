Search.setIndex({envversion:46,filenames:["datasets","detectors","evaluation","features","geometry","index","installation","keypoints","main","pcl","preprocessing","reference","stereo","usage"],objects:{"":{pydriver:[8,0,0,"-"]},"pydriver.datasets":{base:[0,0,0,"-"],kitti:[0,0,0,"-"],utils:[0,0,0,"-"]},"pydriver.datasets.base":{BaseReader:[0,2,1,""],loadImage:[0,4,1,""]},"pydriver.datasets.base.BaseReader":{"__init__":[0,1,1,""],getDatasets:[0,1,1,""],getFrameIds:[0,1,1,""],getFrameInfo:[0,1,1,""],getFramesInfo:[0,1,1,""]},"pydriver.datasets.kitti":{KITTIObjectsReader:[0,2,1,""],KITTIReader:[0,2,1,""],KITTITrackletsReader:[0,2,1,""],correctKITTILabelForStereo:[0,4,1,""],getKITTIGroundTruth:[0,4,1,""],writeLabels:[0,4,1,""]},"pydriver.datasets.kitti.KITTIObjectsReader":{"__init__":[0,1,1,""],getFrameInfo:[0,1,1,""],getFramesInfo:[0,1,1,""]},"pydriver.datasets.kitti.KITTIReader":{"__init__":[0,1,1,""],getFrameInfo:[0,1,1,""],getFramesInfo:[0,1,1,""]},"pydriver.datasets.kitti.KITTITrackletsReader":{getFrameInfo:[0,1,1,""],getFramesInfo:[0,1,1,""]},"pydriver.datasets.utils":{detections2labels:[0,4,1,""],labels2detections:[0,4,1,""]},"pydriver.detectors":{detectors:[1,0,0,"-"],vocabularies:[1,0,0,"-"]},"pydriver.detectors.detectors":{Detector:[1,2,1,""]},"pydriver.detectors.detectors.Detector":{"__init__":[1,1,1,""],addWords:[1,1,1,""],featureTypes:[1,5,1,""],getDetections:[1,1,1,""],learn:[1,1,1,""],load:[1,3,1,""],recognize:[1,1,1,""],save:[1,1,1,""],vocabularyGenerator:[1,5,1,""]},"pydriver.detectors.vocabularies":{Storage:[1,2,1,""],Vocabulary:[1,2,1,""]},"pydriver.detectors.vocabularies.Storage":{"__init__":[1,1,1,""],addWords:[1,1,1,""],category:[1,5,1,""],dims:[1,5,1,""],entries:[1,5,1,""],isEmpty:[1,5,1,""],isPreparedForRecognition:[1,5,1,""],load:[1,3,1,""],prepareForRecognition:[1,1,1,""],preprocessors:[1,5,1,""],recognizeFeatures:[1,1,1,""],regressor:[1,5,1,""],save:[1,1,1,""]},"pydriver.detectors.vocabularies.Vocabulary":{"__init__":[1,1,1,""],addWords:[1,1,1,""],classifier:[1,5,1,""],dims:[1,5,1,""],isEmpty:[1,5,1,""],isPreparedForRecognition:[1,5,1,""],load:[1,3,1,""],prepareForRecognition:[1,1,1,""],preprocessors:[1,5,1,""],recognizeFeatures:[1,1,1,""],save:[1,1,1,""],storageGenerator:[1,5,1,""]},"pydriver.evaluation":{evaluation:[2,0,0,"-"]},"pydriver.evaluation.evaluation":{Evaluator:[2,2,1,""],EvaluatorPoint:[2,2,1,""]},"pydriver.evaluation.evaluation.Evaluator":{"__init__":[2,1,1,""],addFrame:[2,1,1,""],aos:[2,5,1,""],aprecision:[2,5,1,""],getPoint:[2,1,1,""],getPoints:[2,1,1,""],getValues:[2,1,1,""]},"pydriver.evaluation.evaluation.EvaluatorPoint":{"__init__":[2,1,1,""],addFrame:[2,1,1,""],detections:[2,5,1,""],objects:[2,5,1,""],os:[2,5,1,""],precision:[2,5,1,""],recall:[2,5,1,""]},"pydriver.features":{base:[3,0,0,"-"],shot:[3,0,0,"-"]},"pydriver.features.base":{FeatureExtractor:[3,2,1,""]},"pydriver.features.base.FeatureExtractor":{"__init__":[3,5,1,""],dims:[3,5,1,""],getFeatures:[3,1,1,""]},"pydriver.features.shot":{SHOTColorExtractor:[3,2,1,""],SHOTExtractor:[3,2,1,""]},"pydriver.features.shot.SHOTColorExtractor":{"__init__":[3,5,1,""],dims:[3,5,1,""],getFeatures:[3,1,1,""]},"pydriver.features.shot.SHOTExtractor":{"__init__":[3,5,1,""],dims:[3,5,1,""],getFeatures:[3,1,1,""]},"pydriver.geometry":{geometry:[4,0,0,"-"]},"pydriver.geometry.geometry":{affineTransform:[4,4,1,""],cartesian2homogenuous:[4,4,1,""],extractNormalizedOrientedBoxes:[4,4,1,""],get3DBoxVertices:[4,4,1,""],getNormalizationTransformation:[4,4,1,""],homogenuous2cartesian:[4,4,1,""],image2space:[4,4,1,""],project3DBox:[4,4,1,""],transform3DBox:[4,4,1,""]},"pydriver.keypoints":{base:[7,0,0,"-"],harris:[7,0,0,"-"],iss:[7,0,0,"-"]},"pydriver.keypoints.base":{KeypointExtractor:[7,2,1,""],normals2lrfs:[7,4,1,""]},"pydriver.keypoints.base.KeypointExtractor":{"__init__":[7,1,1,""],getKeypointCloud:[7,1,1,""]},"pydriver.keypoints.harris":{HarrisExtractor:[7,2,1,""]},"pydriver.keypoints.harris.HarrisExtractor":{"__init__":[7,1,1,""],getKeypointCloud:[7,1,1,""]},"pydriver.keypoints.iss":{ISSExtractor:[7,2,1,""]},"pydriver.keypoints.iss.ISSExtractor":{"__init__":[7,1,1,""],getKeypointCloud:[7,1,1,""]},"pydriver.pcl":{pcl:[9,0,0,"-"]},"pydriver.pcl.pcl":{PCLHelper:[9,2,1,""],SHOTColorFeature_dtype:[9,6,1,""],SHOTFeature_dtype:[9,6,1,""]},"pydriver.pcl.pcl.PCLHelper":{"__init__":[9,5,1,""],addCloud:[9,1,1,""],computeSHOT:[9,1,1,""],computeSHOTColor:[9,1,1,""],copy:[9,1,1,""],detectGroundPlane:[9,1,1,""],downsampleVoxelGrid:[9,1,1,""],extractOrientedBoxes:[9,1,1,""],fromArray:[9,1,1,""],getCloudSize:[9,1,1,""],getConnectedComponents:[9,1,1,""],getHarrisPoints:[9,1,1,""],getISSPoints:[9,1,1,""],getNormalsOfCloud:[9,1,1,""],isOrganized:[9,1,1,""],removeGroundPlane:[9,1,1,""],removeNaN:[9,1,1,""],restrictViewport:[9,1,1,""],save:[9,1,1,""],setBGColor:[9,1,1,""],setCameraPosition:[9,1,1,""],setDetectionsVisualization:[9,1,1,""],setKeypointsVisualization:[9,1,1,""],setNormalsKSearch:[9,1,1,""],setNormalsRadius:[9,1,1,""],toArray:[9,1,1,""],transform:[9,1,1,""],visualize:[9,1,1,""],visualizeDetections:[9,1,1,""],visualizeKeypoints:[9,1,1,""]},"pydriver.preprocessing":{preprocessing:[10,0,0,"-"]},"pydriver.preprocessing.preprocessing":{CloudProcessor:[10,2,1,""],DownsampleProcessor:[10,2,1,""],GroundPlaneProcessor:[10,2,1,""],LidarReconstructor:[10,2,1,""],Preprocessor:[10,2,1,""],RemoveNaNProcessor:[10,2,1,""],ViewportProcessor:[10,2,1,""]},"pydriver.preprocessing.preprocessing.CloudProcessor":{"__init__":[10,1,1,""],process:[10,1,1,""]},"pydriver.preprocessing.preprocessing.DownsampleProcessor":{"__init__":[10,1,1,""],process:[10,1,1,""]},"pydriver.preprocessing.preprocessing.GroundPlaneProcessor":{"__init__":[10,1,1,""],process:[10,1,1,""]},"pydriver.preprocessing.preprocessing.LidarReconstructor":{"__init__":[10,1,1,""],process:[10,1,1,""]},"pydriver.preprocessing.preprocessing.Preprocessor":{"__init__":[10,1,1,""],process:[10,1,1,""]},"pydriver.preprocessing.preprocessing.RemoveNaNProcessor":{"__init__":[10,1,1,""],process:[10,1,1,""]},"pydriver.preprocessing.preprocessing.ViewportProcessor":{"__init__":[10,1,1,""],process:[10,1,1,""]},"pydriver.stereo":{stereo:[12,0,0,"-"]},"pydriver.stereo.stereo":{OpenCVMatcher:[12,2,1,""],StereoReconstructor:[12,2,1,""],depth2disparity:[12,4,1,""],disparity2depth:[12,4,1,""]},"pydriver.stereo.stereo.OpenCVMatcher":{"__init__":[12,1,1,""],computeDisparity:[12,1,1,""]},"pydriver.stereo.stereo.StereoReconstructor":{"__init__":[12,1,1,""],computeDisparity:[12,1,1,""],process:[12,1,1,""]},pydriver:{Detection_dtype:[8,6,1,""],FLOAT_dtype:[8,6,1,""],Position_dtype:[8,6,1,""],datasets:[0,0,0,"-"],detectors:[1,0,0,"-"],evaluation:[2,0,0,"-"],features:[3,0,0,"-"],geometry:[4,0,0,"-"],keypoints:[7,0,0,"-"],pcl:[9,0,0,"-"],preprocessing:[10,0,0,"-"],stereo:[12,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","method","Python method"],"2":["py","class","Python class"],"3":["py","classmethod","Python class method"],"4":["py","function","Python function"],"5":["py","attribute","Python attribute"],"6":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:method","2":"py:class","3":"py:classmethod","4":"py:function","5":"py:attribute","6":"py:data"},terms:{"4x4":10,"__init__":[0,1,2,3,7,9,10,12],"__pyx_memviewslic":[4,9],"_npoint":2,"abstract":0,"boolean":0,"case":[1,10],"class":[0,1,2,3,7,9,10,12],"default":[0,1,2,3,6,7,9,10,12,13],"final":[6,12],"float":[0,2,3,4,7,8,9,10],"function":[0,1,2,4,9,10,12,13],"import":[6,12,13],"int":[0,1,2,7,9],"new":[1,9,13],"return":[0,1,2,3,4,7,9,10,12,13],"static":6,"switch":6,"true":[0,1,2,4,9,10,13],"try":6,abl:6,about:[0,1,6],abov:[2,13],absenc:13,accept:[1,12],accord:[2,9,10],accordingli:0,adaboostclassifi:[1,13],add:[1,2,6,13],addcloud:9,addend:9,addfram:[2,13],addit:[6,9],addword:[1,13],adjust:[0,10,13],administr:6,affin:[4,9],affinetransform:4,after:[6,10],afterward:1,against:6,aggreg:1,aim:5,algorithm:[0,10],alia:8,all:[0,1,2,4,6,9,12],allow:[6,12],along:[0,7],alpha:0,alreadi:[1,6],also:[0,3,6,12],although:6,alwai:[0,13],ambigu:7,amount:6,angl:[0,7,9,10],anglethreshold:[7,9],ani:[4,9,10,12],anoth:[9,12],anywher:10,append:13,appli:[0,1,4,9,10,13],applic:0,appropri:1,aprecis:[2,13],apt:6,arbitrari:13,archiv:5,area:2,argument:[1,12],around:[0,7,8],arrai:[0,1,3,4,5,7],associ:[1,13],assumpt:2,attent:10,august:6,automat:6,avail:[0,1,2,4,6,9,10],averag:[2,13],avoid:[0,13],awai:8,awar:1,axi:[0,7,8],background:9,balanceneg:[1,13],baseread:[0,1,2,4,9,10],basic:5,batch_siz:13,becom:[0,9],befor:[0,4,7],below:10,benchmark:13,best:4,between:[2,7,8,9,10,13],bgcolor:9,bigger:13,binari:5,bit:6,black:9,block:4,bool:[0,1,3,4,7,9,10],borderradiu:[7,9],both:13,bottom:[0,4,13],bound:[0,2,4,7],boundari:[0,7],box2d:[0,13],box2duntrunc:0,box3d:[0,1,4,13],box3d_exclud:13,box:[0,1,2,4,9,13],boxes3d:9,boxes3d_exclud:13,boxkeypointcloud:13,build:6,build_ext:6,cach:[2,10],calib:0,calibr:[0,10,12,13],call:4,callabl:1,camera:[0,9,10,13],camposit:9,can:[0,1,3,6,9,10,12,13],captur:13,car:[0,13],carcoord:4,cartesian2homogenu:4,cartesian:4,categori:[0,1,2,8,13],categoriesopt:0,center:[0,4],chang:[6,9],channel:[0,12],check:9,christoph:6,classif:[1,5],classifi:[1,5,13],classifict:1,classmethod:1,cloud:[3,6,7,9,10,12,13],cloudprocessor:10,cluster:13,cmake:6,code:[6,13],coeffici:9,color:[3,9,10,13],com:6,combin:10,command:6,compat:6,compens:4,compil:5,complet:6,compon:[6,9],compress:1,comput:[2,3,7,9,12],computedispar:12,computeshot:9,computeshotcolor:9,configur:[1,6,10],consid:[2,9,10,13],construct:9,contain:[0,1,2,4,9,10,13],contribut:13,convers:8,convert:[0,13],convex:2,coordin:[0,3,4,5,7],copi:[4,9,13],core:6,corner:[7,9],correct:[1,13],correctkittilabelforstereo:0,correctli:2,correspond:[1,2,9],count:2,cpu:[4,6],creat:[1,2,9,13],criteria:0,criterion:2,cube:9,current:[1,6,10],curv:2,cv2:12,cyclist:0,cython:6,dark:9,data:[0,1,4,6,8,9,10,12,13],datetim:13,decreas:[2,10],deepcopi:13,def:13,defin:[0,1,9],depend:[0,6,10,13],depth2dispar:12,depth:12,depthmap:12,depthmap_left:12,deriv:[0,10],describ:[3,4,7,10],descript:[0,1,2,4,9,10],descriptor:9,desir:2,destroi:9,detail:7,detect:[0,1,2,5,7,8,9,10,13],detectgroundplan:9,detection_categori:13,detection_categories_opt:13,detection_dtyp:[0,1,8,9],detections2label:[0,13],detections_label:13,detvisparam:9,dev:6,develop:5,devic:4,dict:[0,1,3,4,7,9,10,12],dictionari:[0,1,2,4,9,10,12,13],differ:[1,2,3,10],dim:[1,3,13],dimens:[0,1,3,12,13],dimension:[9,12,13],dircach:10,direct:[8,9],directori:[0,6,10,13],disabl:[3,7,9,10],dismiss:2,dispar:[0,4,12],disparity2depth:12,disparitymap:[4,12],disparitymap_left:12,distanc:[9,10],distancethreshold:9,distinguish:9,distribut:6,document:[3,6,7,10],doe:[2,6,9],don:[10,13],dontcar:0,down:8,download:6,downsampl:[7,9,10],downsampleprocessor:10,downsamplevoxelgrid:9,drawn:2,driver:6,dure:2,each:[0,1,2,9],easi:[0,13],easili:9,edg:9,edit:6,eigenvalu:7,element:[1,9],elemn:1,els:6,empir:0,empti:[1,10],enabl:6,end:0,enough:1,ensembl:[1,13],entri:1,environ:5,equal:[3,12],error:[0,6],estim:[0,7,9],etc:10,evalu:0,evaluation_mod:13,evaluatorpoint:2,everi:[0,6],everyth:2,exact:2,exactli:1,exampl:[1,5],exclud:13,execut:6,exist:6,expect:[0,2,10],experi:5,explicit:10,explicitli:10,extens:6,extract:[0,3,4,6,7,9,10,12,13],extractnormalizedorientedbox:4,extractor:[0,3,7,13],extractorientedbox:[9,13],extractrgb:9,face:0,factor:1,fals:[0,1,2,3,4,9,10,13],far:6,faster:10,featur:1,featuredata:1,featureextractor:[3,13],featurenam:13,featuresdata:1,featuretyp:[1,13],figur:13,file:[0,1,6,9],filenam:[0,9],filepath:0,filter:9,find:13,first:[0,1,3,6,7],firstfram:13,fit:1,fixedi:[3,13],fkeypoint:13,flag:[1,7,9,10],flat:9,flatten:0,float_dtyp:[0,1,3,4,7,8,9,12],float_t:9,follow:[0,4],form:9,format:[0,1,2,4,9,10],frame:[0,2,7,9,10,13],frameid:[0,10,13],framestart:0,framestep:0,framestop:0,from:[0,1,2,3,4,5],fromarrai:9,full:[4,9],fulli:[0,1],fullscreen:9,further:[0,1,7,9],fuse:12,gcc:6,gener:[0,6],get3dboxvertic:4,get:[0,1,2,3,4,6,9,10,13],getclouds:9,getconnectedcompon:9,getdataset:0,getdetect:1,getfeatur:[3,13],getframeid:0,getframeinfo:0,getframesinfo:[0,13],getharrispoint:[7,9],getisspoint:[7,9],getkeypointcloud:[7,13],getkittigroundtruth:[0,13],getnormalizationtransform:4,getnormalsofcloud:9,getpoint:2,getvalu:[2,13],github:6,given:[0,1,2,3,4,7,9,10,12],gohlk:6,good:0,gpu:[4,6],grai:0,grayscal:0,great:6,green:9,grid:[7,9,10],ground:[0,2,9,10,13],groundplaneprocessor:10,groundtruth:[0,2,9,13],groundtruthopt:[0,2,9,13],grow:1,growth_factor:1,gtcolor:9,gtd:13,gtdopt:13,gtoptcolor:9,half:13,hand:8,hard:[0,13],harder:0,hardwar:6,harri:5,harrisextractor:7,harriskeypoint3d:9,have:[0,4,6,7,9,10,12,13],height:[0,8,9,12,13],helper:9,hess:9,homcoord:4,homepag:6,homogenu:4,homogenuous2cartesian:4,how:9,howev:6,http:6,hypothes:[2,9],hypothesi:2,ident:[3,10],ignor:[2,3],imag:[0,4,6,10,12,13],image2spac:4,image_02:0,image_03:0,image_2:0,image_3:0,img_left:[0,10,12,13],img_right:[0,10,12],imgshap:0,implement:[0,1,4,10,12],importerror:12,improv:10,imread:0,includ:[0,3,6,10,13],includealpha:0,includei:3,increas:[2,4],index:5,indic:1,individu:2,info:[0,2,13],inform:[0,1,3,4,6,7,9,10,12],initi:[0,1,2,3,7,9,10,12,13],inlier:[9,10],inplac:[4,10],input:[4,12],insid:[9,13],instal:5,instanc:[1,2,9,10,12],instead:[6,9],instruct:6,integ:0,interact:9,interfac:[0,9],intern:1,interpret:6,intersect:2,interv:2,invalid:[3,9,12],invers:13,invert:[9,13],isempti:1,isorgan:9,ispreparedforrecognit:1,iss:5,issextractor:[7,13],isskeypoint3d:9,issu:6,jochen:6,keep:6,keepdata:1,kei:[0,2,4,9,10],keypoint:[1,3,5],keypointcloud:[3,13],keypointextractor:[7,13],keyword:12,kittiobjectsread:[0,13],kittiread:0,kittitrackletsread:0,kneighborsregressor:[1,13],knn:9,known:6,kpvisparam:9,kwarg:[3,7,10,12],label:[0,2,13],label_02:0,label_2:0,labels2detect:[0,13],larg:[0,6],lastfram:13,later:13,latter:2,launchpad:6,lazi:0,leaf:[7,9,10],leafsiz:[7,9,10],learn:[1,5,6,13],least:[0,1],leav:0,left:[0,4,12,13],legend:13,length:[0,8,9,13],less:3,lesser:2,libpcl:6,librari:[6,8,9],lidar2cam:0,lidar:[0,10,13],lidarreconstructor:[10,13],lie:13,like:[1,6],linearli:2,list:[0,1,2,4,9,13],load:[0,1,10],loadimag:0,loc:13,local:[7,9],locat:[0,4,6],lpltk:6,lrf:7,made:13,mai:[3,6],maintain:6,make:[6,9],makefil:6,mandatori:[0,2,13],mani:6,manual:6,map:[4,12],margin:9,mark:7,mask:9,match:[2,12],matcher:12,matplotlib:[6,13],matric:[0,10,13],matrix:[0,4,9,10,12,13],matter:2,max_it:13,maxangl:10,maxdist:10,maxim:[9,10],maxima:[7,9],maximum:[1,2,4],maxplaneangl:9,mean:[1,4],meaning:9,measur:2,merg:9,meter:10,method:[7,9,12],microsoft:6,min_overlap:13,mind:6,minibatchkmean:13,minim:13,minimum:[2,7],minneighbor:[7,9],minoverlap:[2,13],minweight:2,misc:0,mislead:10,miss:[2,9],mode:[0,6,13],model:1,moder:[0,13],modif:10,modifi:[4,6,10],monoton:2,more:[0,1,6,10],most:6,msi:5,msvc:6,multi:2,multipl:[0,2,6],multipli:1,must:[0,1,2,6,7,9,12],myfeatur:13,n_cluster:13,n_estim:[1,13],n_neighbor:[1,13],name:[0,1,13],nan:[1,4,9,10,12],ndarrai:[0,1,3,4,7,9,12],ndim:[0,1,3,4,7,9,12],need:[6,9,13],neg:[0,1,2,8,9,12,13],negativekeypointcloud:13,negcolor:9,negsiz:9,neighbor:[1,7,13],nmaxrandomsampl:1,non:[0,1,7,9,12,13],none:[0,1,9,10,12],nonmaxradiu:[7,9,13],normal:[2,4,7,9],normallength:9,normalradiu:[7,9],normals2lrf:7,normals_k:9,normals_radiu:9,now:[6,13],npoint:[2,9,13],nrecallinterv:2,nstoragemaxrandomsampl:[1,13],number:[1,2,3,7,9,12],numpi:[0,4,5,6],nvocmaxrandomsampl:1,object:[0,1,2,5,8,9,10,13],observ:0,occlud:0,occlus:0,offer:[6,10],offici:6,onli:[0,1,2,6,7,9,10,13],open:1,opencl:[4,6],opencv:[4,6,12],opencvmatch:12,openmp:6,oper:[6,9,10],optim:4,option:[0,1,2,3,6,7,9,10,12,13],order:[0,4,6,12],organ:[9,10],orient:[2,4,9,13],origin:[3,9,13],other:[1,2,3,6,10,13],otherwis:[1,9],out:[0,12],outdat:6,output:0,outsid:[9,13],overlap:[2,13],overwritten:10,own:[4,12],packag:5,page:5,pai:10,pair:12,parallel:4,paramet:[0,1,2,3,4,7,9,10,12],part:[6,13],partli:0,pass:[10,12],path:[0,13],path_to_kitti:13,pcd:9,pcl:[5,6],pcl_helper:6,pclhelper:[3,7,9,12],pedestrian:0,penal:0,per:1,perform:[1,2,10,13],person_sit:0,perspect:12,pickl:9,pip:6,pipelin:[10,12],pixel:9,planc:9,plane:[7,9,10,13],plot:[2,13],plt:13,point:[2,6,7,8,9,10,12,13],pointcloud:9,portabl:6,portion:10,poscolor:9,posit:[0,1,2,8,9,13],position_dtyp:[1,3,8],possibl:[0,1],ppa:6,pre:6,precis:[2,13],prepar:1,prepareforrecognit:1,preprocess:5,preprocessor:[1,3,7,10,13],presenc:0,present:6,preserv:1,print:13,prior:6,privileg:6,probabl:[1,6],process:[0,2,6,10,12,13],processor:10,produc:[0,1,2,3,10,12],project3dbox:4,project:[0,4,6,7,13],projection_left:[0,10,13],projection_right:[0,10],prompt:6,provid:[0,6,10,12],pydriv:[0,1,2,3,4],pyopencl:6,pyplot:13,python:5,radian:[8,10],radiu:[3,7,9],rais:12,rang:[9,10],ransac:10,ratio:7,reach:0,read:1,reader:[0,13],recal:[2,13],recogn:[1,13],recognit:[1,2,13],recognizefeatur:1,recommend:6,reconstruct:[0,10,12,13],reconstructor:[10,13],rect:0,rectangl:4,rectif:0,redistribut:6,reduc:8,refer:[0,5,7,9],refin:[7,9],reflect:[0,13],region:9,regressor:[1,13],regular:7,relat:13,releas:6,reli:[0,6],reliabl:10,rememb:6,remov:[4,9,10],removegroundplan:9,removeinvis:[10,13],removenan:9,removenanprocessor:10,repositori:6,repres:9,represent:2,reproject:[0,4,12],request:0,requir:[0,1,4,6,9,10,12,13],research:5,resolv:7,respect:[1,2,9],restrict:[0,1,9,10],restrictviewport:9,result:[0,1,2,4,6,7,9,10,12,13],returntyp:[0,1,3,4,7,9,10,12],revert:13,rgb:[0,4,9,10],rgba:0,right:[0,4,8,12,13],road:5,rotat:[0,7,8,9],rotation_i:[0,4,8],roughli:0,run:6,safeti:9,salient:7,salientradiu:[7,9,13],same:[0,1,4,6,12,13],sampl:1,save:[0,1,9],scenario:[2,10],scene:[3,7,10,12,13],scikit:6,scipi:[0,6],screen:9,search:[5,9],second:[1,7],see:[0,1,2,3,4,6,7,9,10,13],segment:[4,9],self:[1,2,3,7,9,10,12],sens:[2,6],sensibl:6,sensor:0,separ:0,sequenc:0,server:0,set:[0,1,9,13],setbgcolor:9,setcameraposit:9,setdetectionsvisu:9,setkeypointsvisu:9,setnormalsksearch:9,setnormalsradiu:9,setup:[6,13],shape:[0,3,6,7,9,12,13],shot_radiu:13,shotcolorextractor:[3,13],shotcolorfeatur:9,shotcolorfeature_dtyp:9,shotextractor:3,shotfeatur:9,shotfeature_dtyp:9,shotradiu:[3,13],should:[0,6],show:13,side:9,sign:9,similar:[2,13],simultan:4,singl:9,size:[0,7,9,10,12],skimag:6,sklearn:[1,6,13],smaller:10,softwar:6,some:[3,6,9,13],someth:6,sort:[0,2],sourc:5,space:[2,4],special:1,specif:[6,12],specifi:[0,1,4,9,12],speed:[4,10],sphinx:6,sprickerhof:6,standalon:5,standard:6,start:6,statu:0,step:[6,13],stereo:[0,5,6,11],stereoreconstructor:[10,12],stereosgbm:12,stick:6,sto:13,stop:0,storag:[1,13],storagegener:[1,13],store:[1,10,13],str:[0,1,8,10],string:[0,1,9,10],structur:[8,9],studio:6,sub:1,subclass:0,submodul:0,sudo:6,suit:13,suitabl:2,suppli:[0,2,6],support:[0,1,6],suppress:[1,7,9],suppressneg:1,surfac:[7,9],system:[0,5,6],test:[6,13],testing_fram:13,than:3,thei:[6,12,13],them:[1,2,6],therefor:6,thi:[0,1,3,4,6,8,9,10,12,13],third:7,those:[2,3,6],three:[0,9],threshold21:[7,9],threshold32:[7,9],threshold:[7,9],thu:6,time:13,timeevalu:13,timelearn:13,timestart:13,timetrain:13,titl:9,toarrai:9,top:[0,4,13],total:9,track:[0,5],trackid:0,tracklet:0,traffic:5,train:[5,6,13],training_fram:13,tram:0,transform3dbox:[4,13],transform:[0,1,4,9,10,13],translat:9,tri:4,truck:0,truncat:[0,13],truth:[0,2,9,13],tupl:[0,1,3,4,9,10,12],type:[1,4,5],ubuntu:5,uint8:[0,4,9],uncach:10,uncompil:6,under:[2,9],union:2,uniqu:[0,1,13],unknown:0,unorgan:9,until:0,updat:[2,6],upgrad:6,upper:[7,13],usabl:6,usag:[5,6],usb:6,use_image_color:13,useimagecolor:[10,13],valid:[1,9,10],validitymask:9,valu:[0,1,2,3,4,7,9,10,12,13],van:[0,13],variou:10,vector:[1,9],velodyn:13,verbos:1,verif:10,version:[6,7],vertex:4,vertic:4,view:9,viewer:8,viewport:[9,10],viewportprocessor:10,visibl:[0,10],vision:13,visual:[1,6,9,13],visualize3d:13,visualizedetect:[9,13],visualizekeypoint:9,voc:13,vocabularygener:[1,13],voxel:[7,9,10],voxelgrid:9,wai:6,want:[2,6,10],weight:[0,1,2,8],well:[0,10],were:[1,13],wheel:5,when:10,where:[0,9],wherea:[10,12],whether:[1,7,9,10],which:[0,1,6,9,10,12,13],whl:6,who:6,whole:13,width:[0,8,9,12,13],window:5,winpython:6,within:[0,9],without:[0,6],won:6,word:1,work:6,write:0,writelabel:0,wrong:10,x64:6,xlabel:13,xlim:13,xyz:[0,4,9,10],xyzrgb:9,yield:0,ylabel:13,ylim:13,you:[2,6,10,13],your:[6,8,10,13]},titles:["The <code class=\"docutils literal\"><span class=\"pre\">datasets</span></code> Module","The <code class=\"docutils literal\"><span class=\"pre\">detectors</span></code> Module","The <code class=\"docutils literal\"><span class=\"pre\">evaluation</span></code> Module","The <code class=\"docutils literal\"><span class=\"pre\">features</span></code> Module","The <code class=\"docutils literal\"><span class=\"pre\">geometry</span></code> Module","PyDriver Framework","Installation","The <code class=\"docutils literal\"><span class=\"pre\">keypoints</span></code> Module","Basics","The <code class=\"docutils literal\"><span class=\"pre\">pcl</span></code> Module","The <code class=\"docutils literal\"><span class=\"pre\">preprocessing</span></code> Module","Reference","The <code class=\"docutils literal\"><span class=\"pre\">stereo</span></code> Module","Usage Example"],titleterms:{archiv:6,arrai:8,base:[0,3,7],basic:8,binari:6,compil:6,coordin:8,dataset:0,detector:1,develop:6,evalu:2,exampl:13,featur:3,framework:5,from:6,geometri:4,harri:7,indic:5,instal:6,iss:7,keypoint:7,kitti:0,modul:[0,1,2,3,4,7,9,10,12],msi:6,numpi:8,packag:6,pcl:9,preprocess:10,pydriv:5,python:6,refer:11,shot:3,sourc:6,standalon:6,stereo:12,system:8,tabl:5,type:8,ubuntu:6,usag:13,util:0,vocabulari:1,wheel:6,window:6}})
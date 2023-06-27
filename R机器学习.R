library(mlr3verse)
dat = tsk("german_credit")$data()
task = as_task_classif(dat, target = "credit_risk")
set.seed(1)
split = partition(task, ratio = 0.7)    # 默认stratify = TRUE, 按目标变量分层

learner = lrn("classif.ranger", num.trees = 100, 
              
              predict_type = "prob")
learner$train(task, row_ids = split$train)
learner$model
prediction = learner$predict(task, row_ids = split$test)
prediction
prediction$score(msr("classif.acc"))      # 准确率
autoplot(prediction, type = "roc")    # 绘制ROC曲线, 需要precrec包
prediction$score(msr("classif.auc"))      # AUC面积


cv10 = rsmp("cv", folds = 10)   # 选择重抽样策略: 10折交叉验证
cv10$instantiate(task)
cv10$iters         # 数据副本数: 10
cv10$train_set(1)   # 第1个数据副本的训练集索引(部分)
rr = resample(task, learner, cv10, store_models = TRUE)
rr$aggregate(msr("classif.acc"))    # 所有重抽样的平均准确率
rr$prediction()        # 所有预测合并为一个预测对象(宏平均)



#benchmark
tasks = tsk("sonar")     # 可以是多个任务
learners = lrns(c("classif.rpart", "classif.kknn", 
                  
                  "classif.ranger", "classif.svm"), 
                
                predict_type = "prob")
design = benchmark_grid(tasks, learners, rsmps("cv", folds = 5))
design 

bmr = benchmark(design)            # 执行基准测试 

# 汇总基准测试结果

bmr$aggregate(list(msr("classif.acc"), msr("classif.auc")))

autoplot(bmr, type = "roc")                  # ROC曲线
autoplot(bmr, measure = msr("classif.auc"))  # AUC箱线图

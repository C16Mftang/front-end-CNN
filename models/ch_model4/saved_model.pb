Їщ
бЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ТЄ


conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
w
x_pred/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*
shared_namex_pred/kernel
p
!x_pred/kernel/Read/ReadVariableOpReadVariableOpx_pred/kernel*
_output_shapes
:	Р*
dtype0
n
x_pred/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namex_pred/bias
g
x_pred/bias/Read/ReadVariableOpReadVariableOpx_pred/bias*
_output_shapes
:*
dtype0
w
y_pred/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*
shared_namey_pred/kernel
p
!y_pred/kernel/Read/ReadVariableOpReadVariableOpy_pred/kernel*
_output_shapes
:	Р*
dtype0
n
y_pred/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namey_pred/bias
g
y_pred/bias/Read/ReadVariableOpReadVariableOpy_pred/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/m

(Adam/conv3d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/m**
_output_shapes
:*
dtype0
|
Adam/conv3d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv3d/bias/m
u
&Adam/conv3d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0

Adam/x_pred/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*%
shared_nameAdam/x_pred/kernel/m
~
(Adam/x_pred/kernel/m/Read/ReadVariableOpReadVariableOpAdam/x_pred/kernel/m*
_output_shapes
:	Р*
dtype0
|
Adam/x_pred/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/x_pred/bias/m
u
&Adam/x_pred/bias/m/Read/ReadVariableOpReadVariableOpAdam/x_pred/bias/m*
_output_shapes
:*
dtype0

Adam/y_pred/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*%
shared_nameAdam/y_pred/kernel/m
~
(Adam/y_pred/kernel/m/Read/ReadVariableOpReadVariableOpAdam/y_pred/kernel/m*
_output_shapes
:	Р*
dtype0
|
Adam/y_pred/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/y_pred/bias/m
u
&Adam/y_pred/bias/m/Read/ReadVariableOpReadVariableOpAdam/y_pred/bias/m*
_output_shapes
:*
dtype0

Adam/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/v

(Adam/conv3d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/v**
_output_shapes
:*
dtype0
|
Adam/conv3d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv3d/bias/v
u
&Adam/conv3d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0

Adam/x_pred/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*%
shared_nameAdam/x_pred/kernel/v
~
(Adam/x_pred/kernel/v/Read/ReadVariableOpReadVariableOpAdam/x_pred/kernel/v*
_output_shapes
:	Р*
dtype0
|
Adam/x_pred/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/x_pred/bias/v
u
&Adam/x_pred/bias/v/Read/ReadVariableOpReadVariableOpAdam/x_pred/bias/v*
_output_shapes
:*
dtype0

Adam/y_pred/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р*%
shared_nameAdam/y_pred/kernel/v
~
(Adam/y_pred/kernel/v/Read/ReadVariableOpReadVariableOpAdam/y_pred/kernel/v*
_output_shapes
:	Р*
dtype0
|
Adam/y_pred/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/y_pred/bias/v
u
&Adam/y_pred/bias/v/Read/ReadVariableOpReadVariableOpAdam/y_pred/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ВP
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*эO
valueуOBрO BйO
Щ
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-3
layer-14
layer_with_weights-4
layer-15
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
 	keras_api
R
!regularization_losses
"trainable_variables
#	variables
$	keras_api
R
%regularization_losses
&trainable_variables
'	variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
R
/regularization_losses
0trainable_variables
1	variables
2	keras_api
R
3regularization_losses
4trainable_variables
5	variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
R
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
h

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
h

Wkernel
Xbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api

]iter

^beta_1

_beta_2
	`decay
alearning_ratemСmТ)mУ*mФ7mХ8mЦQmЧRmШWmЩXmЪvЫvЬ)vЭ*vЮ7vЯ8vаQvбRvвWvгXvд
 
F
0
1
)2
*3
74
85
Q6
R7
W8
X9
F
0
1
)2
*3
74
85
Q6
R7
W8
X9
­
regularization_losses
blayer_metrics
clayer_regularization_losses
trainable_variables

dlayers
emetrics
fnon_trainable_variables
	variables
 
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
glayer_metrics
hlayer_regularization_losses

ilayers
trainable_variables
jmetrics
knon_trainable_variables
	variables
 
 
 
­
regularization_losses
llayer_metrics
mlayer_regularization_losses

nlayers
trainable_variables
ometrics
pnon_trainable_variables
	variables
 
 
 
­
!regularization_losses
qlayer_metrics
rlayer_regularization_losses

slayers
"trainable_variables
tmetrics
unon_trainable_variables
#	variables
 
 
 
­
%regularization_losses
vlayer_metrics
wlayer_regularization_losses

xlayers
&trainable_variables
ymetrics
znon_trainable_variables
'	variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
­
+regularization_losses
{layer_metrics
|layer_regularization_losses

}layers
,trainable_variables
~metrics
non_trainable_variables
-	variables
 
 
 
В
/regularization_losses
layer_metrics
 layer_regularization_losses
layers
0trainable_variables
metrics
non_trainable_variables
1	variables
 
 
 
В
3regularization_losses
layer_metrics
 layer_regularization_losses
layers
4trainable_variables
metrics
non_trainable_variables
5	variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
В
9regularization_losses
layer_metrics
 layer_regularization_losses
layers
:trainable_variables
metrics
non_trainable_variables
;	variables
 
 
 
В
=regularization_losses
layer_metrics
 layer_regularization_losses
layers
>trainable_variables
metrics
non_trainable_variables
?	variables
 
 
 
В
Aregularization_losses
layer_metrics
 layer_regularization_losses
layers
Btrainable_variables
metrics
non_trainable_variables
C	variables
 
 
 
В
Eregularization_losses
layer_metrics
 layer_regularization_losses
layers
Ftrainable_variables
metrics
non_trainable_variables
G	variables
 
 
 
В
Iregularization_losses
layer_metrics
 layer_regularization_losses
 layers
Jtrainable_variables
Ёmetrics
Ђnon_trainable_variables
K	variables
 
 
 
В
Mregularization_losses
Ѓlayer_metrics
 Єlayer_regularization_losses
Ѕlayers
Ntrainable_variables
Іmetrics
Їnon_trainable_variables
O	variables
YW
VARIABLE_VALUEx_pred/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEx_pred/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
В
Sregularization_losses
Јlayer_metrics
 Љlayer_regularization_losses
Њlayers
Ttrainable_variables
Ћmetrics
Ќnon_trainable_variables
U	variables
YW
VARIABLE_VALUEy_pred/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEy_pred/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

W0
X1

W0
X1
В
Yregularization_losses
­layer_metrics
 Ўlayer_regularization_losses
Џlayers
Ztrainable_variables
Аmetrics
Бnon_trainable_variables
[	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15

В0
Г1
Д2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Еtotal

Жcount
З	variables
И	keras_api
8

Йtotal

Кcount
Л	variables
М	keras_api
8

Нtotal

Оcount
П	variables
Р	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Е0
Ж1

З	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

Й0
К1

Л	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

Н0
О1

П	variables
|z
VARIABLE_VALUEAdam/conv3d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/x_pred/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/x_pred/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/y_pred/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/y_pred/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/x_pred/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/x_pred/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/y_pred/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/y_pred/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*4
_output_shapes"
 :џџџџџџџџџ№$@*
dtype0*)
shape :џџџџџџџџџ№$@
э
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d/kernelconv3d/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasy_pred/kernely_pred/biasx_pred/kernelx_pred/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_390162
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ў
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp!x_pred/kernel/Read/ReadVariableOpx_pred/bias/Read/ReadVariableOp!y_pred/kernel/Read/ReadVariableOpy_pred/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp(Adam/conv3d/kernel/m/Read/ReadVariableOp&Adam/conv3d/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp(Adam/x_pred/kernel/m/Read/ReadVariableOp&Adam/x_pred/bias/m/Read/ReadVariableOp(Adam/y_pred/kernel/m/Read/ReadVariableOp&Adam/y_pred/bias/m/Read/ReadVariableOp(Adam/conv3d/kernel/v/Read/ReadVariableOp&Adam/conv3d/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp(Adam/x_pred/kernel/v/Read/ReadVariableOp&Adam/x_pred/bias/v/Read/ReadVariableOp(Adam/y_pred/kernel/v/Read/ReadVariableOp&Adam/y_pred/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_390717
ѕ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasx_pred/kernelx_pred/biasy_pred/kernely_pred/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/conv3d/kernel/mAdam/conv3d/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/x_pred/kernel/mAdam/x_pred/bias/mAdam/y_pred/kernel/mAdam/y_pred/bias/mAdam/conv3d/kernel/vAdam/conv3d/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/x_pred/kernel/vAdam/x_pred/bias/vAdam/y_pred/kernel/vAdam/y_pred/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_390850вы
=
р
H__inference_functional_1_layer_call_and_return_conditional_losses_390100

inputs
conv3d_390063
conv3d_390065
conv2d_390071
conv2d_390073
conv2d_1_390078
conv2d_1_390080
y_pred_390088
y_pred_390090
x_pred_390093
x_pred_390095
identity

identity_1Ђconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂconv3d/StatefulPartitionedCallЂx_pred/StatefulPartitionedCallЂy_pred/StatefulPartitionedCall
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_390063conv3d_390065*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџ№$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_3896752 
conv3d/StatefulPartitionedCall
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_3896302
max_pooling3d/PartitionedCall
dropout/PartitionedCallPartitionedCall&max_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_3897092
dropout/PartitionedCall
#tf_op_layer_Reshape/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_3897282%
#tf_op_layer_Reshape/PartitionedCallЛ
conv2d/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0conv2d_390071conv2d_390073*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3897472 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3896422
max_pooling2d/PartitionedCall
dropout_1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3897812
dropout_1/PartitionedCallЛ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_1_390078conv2d_1_390080*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3898052"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3896542!
max_pooling2d_1/PartitionedCall
dropout_2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_3898392
dropout_2/PartitionedCall
 tf_op_layer_Mean/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_3898582"
 tf_op_layer_Mean/PartitionedCallЅ
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall)tf_op_layer_Mean/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3898722'
%tf_op_layer_Reshape_1/PartitionedCall§
flatten/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3898862
flatten/PartitionedCallЇ
y_pred/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0y_pred_390088y_pred_390090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_y_pred_layer_call_and_return_conditional_losses_3899052 
y_pred/StatefulPartitionedCallЇ
x_pred/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0x_pred_390093x_pred_390095*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_x_pred_layer_call_and_return_conditional_losses_3899322 
x_pred/StatefulPartitionedCallЂ
IdentityIdentity'x_pred/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^conv3d/StatefulPartitionedCall^x_pred/StatefulPartitionedCall^y_pred/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityІ

Identity_1Identity'y_pred/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^conv3d/StatefulPartitionedCall^x_pred/StatefulPartitionedCall^y_pred/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2@
x_pred/StatefulPartitionedCallx_pred/StatefulPartitionedCall2@
y_pred/StatefulPartitionedCally_pred/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
АЋ
Т
"__inference__traced_restore_390850
file_prefix"
assignvariableop_conv3d_kernel"
assignvariableop_1_conv3d_bias$
 assignvariableop_2_conv2d_kernel"
assignvariableop_3_conv2d_bias&
"assignvariableop_4_conv2d_1_kernel$
 assignvariableop_5_conv2d_1_bias$
 assignvariableop_6_x_pred_kernel"
assignvariableop_7_x_pred_bias$
 assignvariableop_8_y_pred_kernel"
assignvariableop_9_y_pred_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1
assignvariableop_19_total_2
assignvariableop_20_count_2,
(assignvariableop_21_adam_conv3d_kernel_m*
&assignvariableop_22_adam_conv3d_bias_m,
(assignvariableop_23_adam_conv2d_kernel_m*
&assignvariableop_24_adam_conv2d_bias_m.
*assignvariableop_25_adam_conv2d_1_kernel_m,
(assignvariableop_26_adam_conv2d_1_bias_m,
(assignvariableop_27_adam_x_pred_kernel_m*
&assignvariableop_28_adam_x_pred_bias_m,
(assignvariableop_29_adam_y_pred_kernel_m*
&assignvariableop_30_adam_y_pred_bias_m,
(assignvariableop_31_adam_conv3d_kernel_v*
&assignvariableop_32_adam_conv3d_bias_v,
(assignvariableop_33_adam_conv2d_kernel_v*
&assignvariableop_34_adam_conv2d_bias_v.
*assignvariableop_35_adam_conv2d_1_kernel_v,
(assignvariableop_36_adam_conv2d_1_bias_v,
(assignvariableop_37_adam_x_pred_kernel_v*
&assignvariableop_38_adam_x_pred_bias_v,
(assignvariableop_39_adam_y_pred_kernel_v*
&assignvariableop_40_adam_y_pred_bias_v
identity_42ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ђ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*ў
valueєBё*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesт
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*О
_output_shapesЋ
Ј::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѓ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ѕ
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѓ
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ѕ
AssignVariableOp_6AssignVariableOp assignvariableop_6_x_pred_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_x_pred_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѕ
AssignVariableOp_8AssignVariableOp assignvariableop_8_y_pred_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ѓ
AssignVariableOp_9AssignVariableOpassignvariableop_9_y_pred_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10Ѕ
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ї
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ї
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13І
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ў
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ё
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ё
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ѓ
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ѓ
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ѓ
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21А
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_conv3d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ў
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_conv3d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23А
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ў
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv2d_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25В
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26А
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27А
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_x_pred_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ў
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_x_pred_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29А
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_y_pred_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ў
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_y_pred_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31А
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv3d_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ў
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv3d_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33А
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ў
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_conv2d_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35В
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36А
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37А
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_x_pred_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ў
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_x_pred_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39А
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_y_pred_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ў
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_y_pred_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpф
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41з
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*Л
_input_shapesЉ
І: :::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_389654

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
г
P
4__inference_tf_op_layer_Reshape_layer_call_fn_390403

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_3897282
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< :[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
У
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_389776

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ	2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

|
'__inference_conv3d_layer_call_fn_390365

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџ№$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_3896752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:џџџџџџџџџ№$@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
Е
M
1__inference_tf_op_layer_Mean_layer_call_fn_390508

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_3898582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с
b
C__inference_dropout_layer_call_and_return_conditional_losses_389704

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeР
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yЪ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< :[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
џ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_389642

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

k
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_389728

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ          2
Reshape/shape
ReshapeReshapeinputsReshape/shape:output:0*
T0*
_cloned(*/
_output_shapes
:џџџџџџџџџ 2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< :[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
З
F
*__inference_dropout_1_layer_call_fn_390450

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3897812
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
 
Њ
B__inference_x_pred_layer_call_and_return_conditional_losses_390541

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
АU

__inference__traced_save_390717
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop,
(savev2_x_pred_kernel_read_readvariableop*
&savev2_x_pred_bias_read_readvariableop,
(savev2_y_pred_kernel_read_readvariableop*
&savev2_y_pred_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop3
/savev2_adam_conv3d_kernel_m_read_readvariableop1
-savev2_adam_conv3d_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop3
/savev2_adam_x_pred_kernel_m_read_readvariableop1
-savev2_adam_x_pred_bias_m_read_readvariableop3
/savev2_adam_y_pred_kernel_m_read_readvariableop1
-savev2_adam_y_pred_bias_m_read_readvariableop3
/savev2_adam_conv3d_kernel_v_read_readvariableop1
-savev2_adam_conv3d_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop3
/savev2_adam_x_pred_kernel_v_read_readvariableop1
-savev2_adam_x_pred_bias_v_read_readvariableop3
/savev2_adam_y_pred_kernel_v_read_readvariableop1
-savev2_adam_y_pred_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5a2425395fff415eb4ef605036cda737/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameь
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*ў
valueєBё*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesм
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesщ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop(savev2_x_pred_kernel_read_readvariableop&savev2_x_pred_bias_read_readvariableop(savev2_y_pred_kernel_read_readvariableop&savev2_y_pred_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop/savev2_adam_conv3d_kernel_m_read_readvariableop-savev2_adam_conv3d_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop/savev2_adam_x_pred_kernel_m_read_readvariableop-savev2_adam_x_pred_bias_m_read_readvariableop/savev2_adam_y_pred_kernel_m_read_readvariableop-savev2_adam_y_pred_bias_m_read_readvariableop/savev2_adam_conv3d_kernel_v_read_readvariableop-savev2_adam_conv3d_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop/savev2_adam_x_pred_kernel_v_read_readvariableop-savev2_adam_x_pred_bias_v_read_readvariableop/savev2_adam_y_pred_kernel_v_read_readvariableop-savev2_adam_y_pred_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*љ
_input_shapesч
ф: :::::::	Р::	Р:: : : : : : : : : : : :::::::	Р::	Р::::::::	Р::	Р:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	Р: 

_output_shapes
::%	!

_output_shapes
:	Р: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	Р: 

_output_shapes
::%!

_output_shapes
:	Р: 

_output_shapes
::0 ,
*
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::%&!

_output_shapes
:	Р: '

_output_shapes
::%(!

_output_shapes
:	Р: )

_output_shapes
::*

_output_shapes
: 
 
Њ
B__inference_y_pred_layer_call_and_return_conditional_losses_389905

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
§
Њ
B__inference_conv2d_layer_call_and_return_conditional_losses_390414

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Tanhd
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ :::W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ЊW
Ў
H__inference_functional_1_layer_call_and_return_conditional_losses_390237

inputs)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource)
%y_pred_matmul_readvariableop_resource*
&y_pred_biasadd_readvariableop_resource)
%x_pred_matmul_readvariableop_resource*
&x_pred_biasadd_readvariableop_resource
identity

identity_1Ў
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02
conv3d/Conv3D/ReadVariableOpО
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@*
paddingSAME*
strides	
2
conv3d/Conv3DЁ
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv3d/BiasAdd/ReadVariableOpЉ
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2
conv3d/BiasAddz
conv3d/TanhTanhconv3d/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2
conv3d/TanhЬ
max_pooling3d/MaxPool3D	MaxPool3Dconv3d/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџ< *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d/MaxPool3Ds
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/ConstБ
dropout/dropout/MulMul max_pooling3d/MaxPool3D:output:0dropout/dropout/Const:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/dropout/Mul~
dropout/dropout/ShapeShape max_pooling3d/MaxPool3D:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeи
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< *
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2 
dropout/dropout/GreaterEqual/yъ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/dropout/GreaterEqualЃ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/dropout/CastІ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/dropout/Mul_1
!tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ          2#
!tf_op_layer_Reshape/Reshape/shapeе
tf_op_layer_Reshape/ReshapeReshapedropout/dropout/Mul_1:z:0*tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*/
_output_shapes
:џџџџџџџџџ 2
tf_op_layer_Reshape/ReshapeЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpж
conv2d/Conv2DConv2D$tf_op_layer_Reshape/Reshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d/BiasAddu
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d/TanhЗ
max_pooling2d/MaxPoolMaxPoolconv2d/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ	*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/dropout/ConstБ
dropout_1/dropout/MulMulmax_pooling2d/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeк
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2"
 dropout_1/dropout/GreaterEqual/yю
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2 
dropout_1/dropout/GreaterEqualЅ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ	2
dropout_1/dropout/CastЊ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout_1/dropout/Mul_1А
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpг
conv2d_1/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
conv2d_1/BiasAdd{
conv2d_1/TanhTanhconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
conv2d_1/TanhН
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/dropout/ConstГ
dropout_2/dropout/MulMul max_pooling2d_1/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeк
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2"
 dropout_2/dropout/GreaterEqual/yю
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2 
dropout_2/dropout/GreaterEqualЅ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout_2/dropout/CastЊ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_2/dropout/Mul_1Ѓ
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2)
'tf_op_layer_Mean/Mean/reduction_indicesЦ
tf_op_layer_Mean/MeanMeandropout_2/dropout/Mul_1:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mean/MeanЃ
%tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ<      2'
%tf_op_layer_Reshape_1/Reshape_1/shapeт
tf_op_layer_Reshape_1/Reshape_1Reshapetf_op_layer_Mean/Mean:output:0.tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ<2!
tf_op_layer_Reshape_1/Reshape_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџР  2
flatten/ConstЂ
flatten/ReshapeReshape(tf_op_layer_Reshape_1/Reshape_1:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/ReshapeЃ
y_pred/MatMul/ReadVariableOpReadVariableOp%y_pred_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
y_pred/MatMul/ReadVariableOp
y_pred/MatMulMatMulflatten/Reshape:output:0$y_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
y_pred/MatMulЁ
y_pred/BiasAdd/ReadVariableOpReadVariableOp&y_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
y_pred/BiasAdd/ReadVariableOp
y_pred/BiasAddBiasAddy_pred/MatMul:product:0%y_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
y_pred/BiasAddm
y_pred/TanhTanhy_pred/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
y_pred/TanhЃ
x_pred/MatMul/ReadVariableOpReadVariableOp%x_pred_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
x_pred/MatMul/ReadVariableOp
x_pred/MatMulMatMulflatten/Reshape:output:0$x_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
x_pred/MatMulЁ
x_pred/BiasAdd/ReadVariableOpReadVariableOp&x_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
x_pred/BiasAdd/ReadVariableOp
x_pred/BiasAddBiasAddx_pred/MatMul:product:0%x_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
x_pred/BiasAddm
x_pred/TanhTanhx_pred/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
x_pred/Tanhc
IdentityIdentityx_pred/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityg

Identity_1Identityy_pred/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@:::::::::::\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
н
|
'__inference_x_pred_layer_call_fn_390550

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_x_pred_layer_call_and_return_conditional_losses_3899322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
У
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_389834

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
У
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_390482

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

k
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_390398

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ          2
Reshape/shape
ReshapeReshapeinputsReshape/shape:output:0*
T0*
_cloned(*/
_output_shapes
:џџџџџџџџџ 2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< :[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
У
D
(__inference_dropout_layer_call_fn_390392

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_3897092
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< :[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
ш
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_390440

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
Ќ
J
.__inference_max_pooling2d_layer_call_fn_389648

inputs
identityэ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3896422
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
с
b
C__inference_dropout_layer_call_and_return_conditional_losses_390377

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeР
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yЪ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/Mul_1q
IdentityIdentitydropout/Mul_1:z:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< :[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
У
c
*__inference_dropout_1_layer_call_fn_390445

inputs
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3897762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
 	
Њ
B__inference_conv3d_layer_call_and_return_conditional_losses_389675

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOpЉ
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@*
paddingSAME*
strides	
2
Conv3D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2	
BiasAdde
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2
Tanhi
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:џџџџџџџџџ№$@:::\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
і
a
C__inference_dropout_layer_call_and_return_conditional_losses_389709

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:џџџџџџџџџ< 2

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2

Identity_1"!

identity_1Identity_1:output:0*2
_input_shapes!
:џџџџџџџџџ< :[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
ї
m
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_389872

inputs
identityw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ<      2
Reshape_1/shape
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ<2
	Reshape_1j
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б


$__inference_signature_wrapper_390162
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1ЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_3896242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџ№$@
!
_user_specified_name	input_1
н
|
'__inference_y_pred_layer_call_fn_390570

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_y_pred_layer_call_and_return_conditional_losses_3899052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
џ
Ќ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_390461

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
Tanhd
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ	:::W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
=
с
H__inference_functional_1_layer_call_and_return_conditional_losses_389990
input_1
conv3d_389953
conv3d_389955
conv2d_389961
conv2d_389963
conv2d_1_389968
conv2d_1_389970
y_pred_389978
y_pred_389980
x_pred_389983
x_pred_389985
identity

identity_1Ђconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂconv3d/StatefulPartitionedCallЂx_pred/StatefulPartitionedCallЂy_pred/StatefulPartitionedCall
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_389953conv3d_389955*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџ№$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_3896752 
conv3d/StatefulPartitionedCall
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_3896302
max_pooling3d/PartitionedCall
dropout/PartitionedCallPartitionedCall&max_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_3897092
dropout/PartitionedCall
#tf_op_layer_Reshape/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_3897282%
#tf_op_layer_Reshape/PartitionedCallЛ
conv2d/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0conv2d_389961conv2d_389963*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3897472 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3896422
max_pooling2d/PartitionedCall
dropout_1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3897812
dropout_1/PartitionedCallЛ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_1_389968conv2d_1_389970*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3898052"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3896542!
max_pooling2d_1/PartitionedCall
dropout_2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_3898392
dropout_2/PartitionedCall
 tf_op_layer_Mean/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_3898582"
 tf_op_layer_Mean/PartitionedCallЅ
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall)tf_op_layer_Mean/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3898722'
%tf_op_layer_Reshape_1/PartitionedCall§
flatten/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3898862
flatten/PartitionedCallЇ
y_pred/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0y_pred_389978y_pred_389980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_y_pred_layer_call_and_return_conditional_losses_3899052 
y_pred/StatefulPartitionedCallЇ
x_pred/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0x_pred_389983x_pred_389985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_x_pred_layer_call_and_return_conditional_losses_3899322 
x_pred/StatefulPartitionedCallЂ
IdentityIdentity'x_pred/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^conv3d/StatefulPartitionedCall^x_pred/StatefulPartitionedCall^y_pred/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityІ

Identity_1Identity'y_pred/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^conv3d/StatefulPartitionedCall^x_pred/StatefulPartitionedCall^y_pred/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2@
x_pred/StatefulPartitionedCallx_pred/StatefulPartitionedCall2@
y_pred/StatefulPartitionedCally_pred/StatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџ№$@
!
_user_specified_name	input_1
 
Њ
B__inference_x_pred_layer_call_and_return_conditional_losses_389932

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
У
c
*__inference_dropout_2_layer_call_fn_390492

inputs
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_3898342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_389781

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
У
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_390435

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeМ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yЦ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:џџџџџџџџџ	2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
р
J
.__inference_max_pooling3d_layer_call_fn_389636

inputs
identityњ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_3896302
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
e
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_389630

inputs
identityЫ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ: {
W
_output_shapesE
C:Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
_
C__inference_flatten_layer_call_and_return_conditional_losses_390525

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџР  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ<:S O
+
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
о


-__inference_functional_1_layer_call_fn_390345

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1ЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_3901002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
і
a
C__inference_dropout_layer_call_and_return_conditional_losses_390382

inputs

identity_1f
IdentityIdentityinputs*
T0*3
_output_shapes!
:џџџџџџџџџ< 2

Identityu

Identity_1IdentityIdentity:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2

Identity_1"!

identity_1Identity_1:output:0*2
_input_shapes!
:џџџџџџџџџ< :[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
§
Њ
B__inference_conv2d_layer_call_and_return_conditional_losses_389747

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Tanhd
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ :::W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
З
F
*__inference_dropout_2_layer_call_fn_390497

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_3898392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ХB
Ъ
H__inference_functional_1_layer_call_and_return_conditional_losses_390033

inputs
conv3d_389996
conv3d_389998
conv2d_390004
conv2d_390006
conv2d_1_390011
conv2d_1_390013
y_pred_390021
y_pred_390023
x_pred_390026
x_pred_390028
identity

identity_1Ђconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂconv3d/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂx_pred/StatefulPartitionedCallЂy_pred/StatefulPartitionedCall
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_389996conv3d_389998*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџ№$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_3896752 
conv3d/StatefulPartitionedCall
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_3896302
max_pooling3d/PartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_3897042!
dropout/StatefulPartitionedCallЂ
#tf_op_layer_Reshape/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_3897282%
#tf_op_layer_Reshape/PartitionedCallЛ
conv2d/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0conv2d_390004conv2d_390006*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3897472 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3896422
max_pooling2d/PartitionedCallМ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3897762#
!dropout_1/StatefulPartitionedCallУ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_1_390011conv2d_1_390013*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3898052"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3896542!
max_pooling2d_1/PartitionedCallР
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_3898342#
!dropout_2/StatefulPartitionedCall
 tf_op_layer_Mean/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_3898582"
 tf_op_layer_Mean/PartitionedCallЅ
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall)tf_op_layer_Mean/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3898722'
%tf_op_layer_Reshape_1/PartitionedCall§
flatten/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3898862
flatten/PartitionedCallЇ
y_pred/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0y_pred_390021y_pred_390023*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_y_pred_layer_call_and_return_conditional_losses_3899052 
y_pred/StatefulPartitionedCallЇ
x_pred/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0x_pred_390026x_pred_390028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_x_pred_layer_call_and_return_conditional_losses_3899322 
x_pred/StatefulPartitionedCall
IdentityIdentity'x_pred/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^conv3d/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^x_pred/StatefulPartitionedCall^y_pred/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity'y_pred/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^conv3d/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^x_pred/StatefulPartitionedCall^y_pred/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2@
x_pred/StatefulPartitionedCallx_pred/StatefulPartitionedCall2@
y_pred/StatefulPartitionedCally_pred/StatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
ш
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_390487

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з:
Ў
H__inference_functional_1_layer_call_and_return_conditional_losses_390291

inputs)
%conv3d_conv3d_readvariableop_resource*
&conv3d_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource)
%y_pred_matmul_readvariableop_resource*
&y_pred_biasadd_readvariableop_resource)
%x_pred_matmul_readvariableop_resource*
&x_pred_biasadd_readvariableop_resource
identity

identity_1Ў
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02
conv3d/Conv3D/ReadVariableOpО
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@*
paddingSAME*
strides	
2
conv3d/Conv3DЁ
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv3d/BiasAdd/ReadVariableOpЉ
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2
conv3d/BiasAddz
conv3d/TanhTanhconv3d/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2
conv3d/TanhЬ
max_pooling3d/MaxPool3D	MaxPool3Dconv3d/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџ< *
ksize	
*
paddingVALID*
strides	
2
max_pooling3d/MaxPool3D
dropout/IdentityIdentity max_pooling3d/MaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
dropout/Identity
!tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ          2#
!tf_op_layer_Reshape/Reshape/shapeе
tf_op_layer_Reshape/ReshapeReshapedropout/Identity:output:0*tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*/
_output_shapes
:џџџџџџџџџ 2
tf_op_layer_Reshape/ReshapeЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpж
conv2d/Conv2DConv2D$tf_op_layer_Reshape/Reshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЄ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d/BiasAddu
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv2d/TanhЗ
max_pooling2d/MaxPoolMaxPoolconv2d/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ	*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
dropout_1/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
dropout_1/IdentityА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpг
conv2d_1/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
paddingSAME*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
conv2d_1/BiasAdd{
conv2d_1/TanhTanhconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
conv2d_1/TanhН
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool
dropout_2/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
dropout_2/IdentityЃ
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2)
'tf_op_layer_Mean/Mean/reduction_indicesЦ
tf_op_layer_Mean/MeanMeandropout_2/Identity:output:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mean/MeanЃ
%tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ<      2'
%tf_op_layer_Reshape_1/Reshape_1/shapeт
tf_op_layer_Reshape_1/Reshape_1Reshapetf_op_layer_Mean/Mean:output:0.tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ<2!
tf_op_layer_Reshape_1/Reshape_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџР  2
flatten/ConstЂ
flatten/ReshapeReshape(tf_op_layer_Reshape_1/Reshape_1:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/ReshapeЃ
y_pred/MatMul/ReadVariableOpReadVariableOp%y_pred_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
y_pred/MatMul/ReadVariableOp
y_pred/MatMulMatMulflatten/Reshape:output:0$y_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
y_pred/MatMulЁ
y_pred/BiasAdd/ReadVariableOpReadVariableOp&y_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
y_pred/BiasAdd/ReadVariableOp
y_pred/BiasAddBiasAddy_pred/MatMul:product:0%y_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
y_pred/BiasAddm
y_pred/TanhTanhy_pred/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
y_pred/TanhЃ
x_pred/MatMul/ReadVariableOpReadVariableOp%x_pred_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
x_pred/MatMul/ReadVariableOp
x_pred/MatMulMatMulflatten/Reshape:output:0$x_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
x_pred/MatMulЁ
x_pred/BiasAdd/ReadVariableOpReadVariableOp&x_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
x_pred/BiasAdd/ReadVariableOp
x_pred/BiasAddBiasAddx_pred/MatMul:product:0%x_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
x_pred/BiasAddm
x_pred/TanhTanhx_pred/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
x_pred/Tanhc
IdentityIdentityx_pred/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityg

Identity_1Identityy_pred/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@:::::::::::\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
ш
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_389839

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ
h
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_389858

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indices~
MeanMeaninputsMean/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с


-__inference_functional_1_layer_call_fn_390125
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1ЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_3901002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџ№$@
!
_user_specified_name	input_1
Г
_
C__inference_flatten_layer_call_and_return_conditional_losses_389886

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџР  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ<:S O
+
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
љ
h
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_390503

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indices~
MeanMeaninputsMean/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЛH

!__inference__wrapped_model_389624
input_16
2functional_1_conv3d_conv3d_readvariableop_resource7
3functional_1_conv3d_biasadd_readvariableop_resource6
2functional_1_conv2d_conv2d_readvariableop_resource7
3functional_1_conv2d_biasadd_readvariableop_resource8
4functional_1_conv2d_1_conv2d_readvariableop_resource9
5functional_1_conv2d_1_biasadd_readvariableop_resource6
2functional_1_y_pred_matmul_readvariableop_resource7
3functional_1_y_pred_biasadd_readvariableop_resource6
2functional_1_x_pred_matmul_readvariableop_resource7
3functional_1_x_pred_biasadd_readvariableop_resource
identity

identity_1е
)functional_1/conv3d/Conv3D/ReadVariableOpReadVariableOp2functional_1_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02+
)functional_1/conv3d/Conv3D/ReadVariableOpц
functional_1/conv3d/Conv3DConv3Dinput_11functional_1/conv3d/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@*
paddingSAME*
strides	
2
functional_1/conv3d/Conv3DШ
*functional_1/conv3d/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/conv3d/BiasAdd/ReadVariableOpн
functional_1/conv3d/BiasAddBiasAdd#functional_1/conv3d/Conv3D:output:02functional_1/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2
functional_1/conv3d/BiasAddЁ
functional_1/conv3d/TanhTanh$functional_1/conv3d/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2
functional_1/conv3d/Tanhѓ
$functional_1/max_pooling3d/MaxPool3D	MaxPool3Dfunctional_1/conv3d/Tanh:y:0*
T0*3
_output_shapes!
:џџџџџџџџџ< *
ksize	
*
paddingVALID*
strides	
2&
$functional_1/max_pooling3d/MaxPool3DЗ
functional_1/dropout/IdentityIdentity-functional_1/max_pooling3d/MaxPool3D:output:0*
T0*3
_output_shapes!
:џџџџџџџџџ< 2
functional_1/dropout/IdentityЙ
.functional_1/tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ          20
.functional_1/tf_op_layer_Reshape/Reshape/shape
(functional_1/tf_op_layer_Reshape/ReshapeReshape&functional_1/dropout/Identity:output:07functional_1/tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*/
_output_shapes
:џџџџџџџџџ 2*
(functional_1/tf_op_layer_Reshape/Reshapeб
)functional_1/conv2d/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)functional_1/conv2d/Conv2D/ReadVariableOp
functional_1/conv2d/Conv2DConv2D1functional_1/tf_op_layer_Reshape/Reshape:output:01functional_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
functional_1/conv2d/Conv2DШ
*functional_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/conv2d/BiasAdd/ReadVariableOpи
functional_1/conv2d/BiasAddBiasAdd#functional_1/conv2d/Conv2D:output:02functional_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
functional_1/conv2d/BiasAdd
functional_1/conv2d/TanhTanh$functional_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
functional_1/conv2d/Tanhо
"functional_1/max_pooling2d/MaxPoolMaxPoolfunctional_1/conv2d/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ	*
ksize
*
paddingVALID*
strides
2$
"functional_1/max_pooling2d/MaxPoolЕ
functional_1/dropout_1/IdentityIdentity+functional_1/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2!
functional_1/dropout_1/Identityз
+functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+functional_1/conv2d_1/Conv2D/ReadVariableOp
functional_1/conv2d_1/Conv2DConv2D(functional_1/dropout_1/Identity:output:03functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
paddingSAME*
strides
2
functional_1/conv2d_1/Conv2DЮ
,functional_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,functional_1/conv2d_1/BiasAdd/ReadVariableOpр
functional_1/conv2d_1/BiasAddBiasAdd%functional_1/conv2d_1/Conv2D:output:04functional_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
functional_1/conv2d_1/BiasAddЂ
functional_1/conv2d_1/TanhTanh&functional_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
functional_1/conv2d_1/Tanhф
$functional_1/max_pooling2d_1/MaxPoolMaxPoolfunctional_1/conv2d_1/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_1/MaxPoolЗ
functional_1/dropout_2/IdentityIdentity-functional_1/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2!
functional_1/dropout_2/IdentityН
4functional_1/tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      26
4functional_1/tf_op_layer_Mean/Mean/reduction_indicesњ
"functional_1/tf_op_layer_Mean/MeanMean(functional_1/dropout_2/Identity:output:0=functional_1/tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2$
"functional_1/tf_op_layer_Mean/MeanН
2functional_1/tf_op_layer_Reshape_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ<      24
2functional_1/tf_op_layer_Reshape_1/Reshape_1/shape
,functional_1/tf_op_layer_Reshape_1/Reshape_1Reshape+functional_1/tf_op_layer_Mean/Mean:output:0;functional_1/tf_op_layer_Reshape_1/Reshape_1/shape:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ<2.
,functional_1/tf_op_layer_Reshape_1/Reshape_1
functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџР  2
functional_1/flatten/Constж
functional_1/flatten/ReshapeReshape5functional_1/tf_op_layer_Reshape_1/Reshape_1:output:0#functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
functional_1/flatten/ReshapeЪ
)functional_1/y_pred/MatMul/ReadVariableOpReadVariableOp2functional_1_y_pred_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02+
)functional_1/y_pred/MatMul/ReadVariableOpЮ
functional_1/y_pred/MatMulMatMul%functional_1/flatten/Reshape:output:01functional_1/y_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/y_pred/MatMulШ
*functional_1/y_pred/BiasAdd/ReadVariableOpReadVariableOp3functional_1_y_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/y_pred/BiasAdd/ReadVariableOpб
functional_1/y_pred/BiasAddBiasAdd$functional_1/y_pred/MatMul:product:02functional_1/y_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/y_pred/BiasAdd
functional_1/y_pred/TanhTanh$functional_1/y_pred/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/y_pred/TanhЪ
)functional_1/x_pred/MatMul/ReadVariableOpReadVariableOp2functional_1_x_pred_matmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02+
)functional_1/x_pred/MatMul/ReadVariableOpЮ
functional_1/x_pred/MatMulMatMul%functional_1/flatten/Reshape:output:01functional_1/x_pred/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/x_pred/MatMulШ
*functional_1/x_pred/BiasAdd/ReadVariableOpReadVariableOp3functional_1_x_pred_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/x_pred/BiasAdd/ReadVariableOpб
functional_1/x_pred/BiasAddBiasAdd$functional_1/x_pred/MatMul:product:02functional_1/x_pred/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/x_pred/BiasAdd
functional_1/x_pred/TanhTanh$functional_1/x_pred/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/x_pred/Tanhp
IdentityIdentityfunctional_1/x_pred/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityt

Identity_1Identityfunctional_1/y_pred/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@:::::::::::] Y
4
_output_shapes"
 :џџџџџџџџџ№$@
!
_user_specified_name	input_1
џ
~
)__inference_conv2d_1_layer_call_fn_390470

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3898052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ	::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

D
(__inference_flatten_layer_call_fn_390530

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3898862
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ<:S O
+
_output_shapes
:џџџџџџџџџ<
 
_user_specified_nameinputs
 
Њ
B__inference_y_pred_layer_call_and_return_conditional_losses_390561

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
о


-__inference_functional_1_layer_call_fn_390318

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1ЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_3900332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
ћ
|
'__inference_conv2d_layer_call_fn_390423

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3897472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Я
a
(__inference_dropout_layer_call_fn_390387

inputs
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_3897042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:џџџџџџџџџ< 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ< 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:џџџџџџџџџ< 
 
_user_specified_nameinputs
ШB
Ы
H__inference_functional_1_layer_call_and_return_conditional_losses_389950
input_1
conv3d_389686
conv3d_389688
conv2d_389758
conv2d_389760
conv2d_1_389816
conv2d_1_389818
y_pred_389916
y_pred_389918
x_pred_389943
x_pred_389945
identity

identity_1Ђconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂconv3d/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂx_pred/StatefulPartitionedCallЂy_pred/StatefulPartitionedCall
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_389686conv3d_389688*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџ№$@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_3896752 
conv3d/StatefulPartitionedCall
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_3896302
max_pooling3d/PartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:џџџџџџџџџ< * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_3897042!
dropout/StatefulPartitionedCallЂ
#tf_op_layer_Reshape/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_3897282%
#tf_op_layer_Reshape/PartitionedCallЛ
conv2d/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0conv2d_389758conv2d_389760*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3897472 
conv2d/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3896422
max_pooling2d/PartitionedCallМ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_3897762#
!dropout_1/StatefulPartitionedCallУ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_1_389816conv2d_1_389818*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3898052"
 conv2d_1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3896542!
max_pooling2d_1/PartitionedCallР
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_3898342#
!dropout_2/StatefulPartitionedCall
 tf_op_layer_Mean/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_3898582"
 tf_op_layer_Mean/PartitionedCallЅ
%tf_op_layer_Reshape_1/PartitionedCallPartitionedCall)tf_op_layer_Mean/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3898722'
%tf_op_layer_Reshape_1/PartitionedCall§
flatten/PartitionedCallPartitionedCall.tf_op_layer_Reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3898862
flatten/PartitionedCallЇ
y_pred/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0y_pred_389916y_pred_389918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_y_pred_layer_call_and_return_conditional_losses_3899052 
y_pred/StatefulPartitionedCallЇ
x_pred/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0x_pred_389943x_pred_389945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_x_pred_layer_call_and_return_conditional_losses_3899322 
x_pred/StatefulPartitionedCall
IdentityIdentity'x_pred/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^conv3d/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^x_pred/StatefulPartitionedCall^y_pred/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity'y_pred/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^conv3d/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^x_pred/StatefulPartitionedCall^y_pred/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2@
x_pred/StatefulPartitionedCallx_pred/StatefulPartitionedCall2@
y_pred/StatefulPartitionedCally_pred/StatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџ№$@
!
_user_specified_name	input_1
ї
m
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_390514

inputs
identityw
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ<      2
Reshape_1/shape
	Reshape_1ReshapeinputsReshape_1/shape:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ<2
	Reshape_1j
IdentityIdentityReshape_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А
L
0__inference_max_pooling2d_1_layer_call_fn_389660

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3896542
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џ
Ќ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_389805

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
Tanhd
IdentityIdentityTanh:y:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ	:::W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
 	
Њ
B__inference_conv3d_layer_call_and_return_conditional_losses_390356

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOpЉ
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@*
paddingSAME*
strides	
2
Conv3D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2	
BiasAdde
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2
Tanhi
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџ№$@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:џџџџџџџџџ№$@:::\ X
4
_output_shapes"
 :џџџџџџџџџ№$@
 
_user_specified_nameinputs
З
R
6__inference_tf_op_layer_Reshape_1_layer_call_fn_390519

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_3898722
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ<2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с


-__inference_functional_1_layer_call_fn_390058
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1ЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_3900332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*[
_input_shapesJ
H:џџџџџџџџџ№$@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :џџџџџџџџџ№$@
!
_user_specified_name	input_1"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ђ
serving_defaultо
H
input_1=
serving_default_input_1:0џџџџџџџџџ№$@:
x_pred0
StatefulPartitionedCall:0џџџџџџџџџ:
y_pred0
StatefulPartitionedCall:1џџџџџџџџџtensorflow/serving/predict:хМ
Ѓr
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-3
layer-14
layer_with_weights-4
layer-15
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+е&call_and_return_all_conditional_losses
ж_default_save_signature
з__call__"§m
_tf_keras_networkсm{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 36, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [6, 3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 2, 2]}, "data_format": "channels_last"}, "name": "max_pooling3d", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling3d", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["dropout/cond/Identity", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 18, 32, 8]}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["dropout_2/cond/Identity", "Mean/reduction_indices"], "attr": {"T": {"type": "DT_FLOAT"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [1, 2]}}, "name": "tf_op_layer_Mean", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["Mean", "Reshape_1/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-1, 60, 16]}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["tf_op_layer_Mean", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["tf_op_layer_Reshape_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "x_pred", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "x_pred", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "y_pred", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "y_pred", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["x_pred", 0, 0], ["y_pred", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 240, 36, 64, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 36, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [6, 3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 2, 2]}, "data_format": "channels_last"}, "name": "max_pooling3d", "inbound_nodes": [[["conv3d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling3d", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["dropout/cond/Identity", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 18, 32, 8]}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["dropout_2/cond/Identity", "Mean/reduction_indices"], "attr": {"T": {"type": "DT_FLOAT"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [1, 2]}}, "name": "tf_op_layer_Mean", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["Mean", "Reshape_1/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-1, 60, 16]}}, "name": "tf_op_layer_Reshape_1", "inbound_nodes": [[["tf_op_layer_Mean", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["tf_op_layer_Reshape_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "x_pred", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "x_pred", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "y_pred", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "y_pred", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["x_pred", 0, 0], ["y_pred", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"
_tf_keras_input_layerр{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 36, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 240, 36, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ћ	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+и&call_and_return_all_conditional_losses
й__call__"д
_tf_keras_layerК{"class_name": "Conv3D", "name": "conv3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [6, 3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 240, 36, 64, 3]}}

regularization_losses
trainable_variables
	variables
 	keras_api
+к&call_and_return_all_conditional_losses
л__call__"ђ
_tf_keras_layerи{"class_name": "MaxPooling3D", "name": "max_pooling3d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling3d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
у
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+м&call_and_return_all_conditional_losses
н__call__"в
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}

%regularization_losses
&trainable_variables
'	variables
(	keras_api
+о&call_and_return_all_conditional_losses
п__call__"
_tf_keras_layer№{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["dropout/cond/Identity", "Reshape/shape"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {"1": [-1, 18, 32, 8]}}}
ю	

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+р&call_and_return_all_conditional_losses
с__call__"Ч
_tf_keras_layer­{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 32, 8]}}
§
/regularization_losses
0trainable_variables
1	variables
2	keras_api
+т&call_and_return_all_conditional_losses
у__call__"ь
_tf_keras_layerв{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ч
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ѓ	

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"Ь
_tf_keras_layerВ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 16, 16]}}

=regularization_losses
>trainable_variables
?	variables
@	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"№
_tf_keras_layerж{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ч
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+ъ&call_and_return_all_conditional_losses
ы__call__"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Њ
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+ь&call_and_return_all_conditional_losses
э__call__"
_tf_keras_layerџ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["dropout_2/cond/Identity", "Mean/reduction_indices"], "attr": {"T": {"type": "DT_FLOAT"}, "Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}}}, "constants": {"1": [1, 2]}}}

Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+ю&call_and_return_all_conditional_losses
я__call__"ў
_tf_keras_layerф{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Reshape_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape_1", "op": "Reshape", "input": ["Mean", "Reshape_1/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-1, 60, 16]}}}
ф
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+№&call_and_return_all_conditional_losses
ё__call__"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ё

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+ђ&call_and_return_all_conditional_losses
ѓ__call__"Ъ
_tf_keras_layerА{"class_name": "Dense", "name": "x_pred", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "x_pred", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 960}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 960]}}
ё

Wkernel
Xbias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+є&call_and_return_all_conditional_losses
ѕ__call__"Ъ
_tf_keras_layerА{"class_name": "Dense", "name": "y_pred", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "y_pred", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 960}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 960]}}

]iter

^beta_1

_beta_2
	`decay
alearning_ratemСmТ)mУ*mФ7mХ8mЦQmЧRmШWmЩXmЪvЫvЬ)vЭ*vЮ7vЯ8vаQvбRvвWvгXvд"
	optimizer
 "
trackable_list_wrapper
f
0
1
)2
*3
74
85
Q6
R7
W8
X9"
trackable_list_wrapper
f
0
1
)2
*3
74
85
Q6
R7
W8
X9"
trackable_list_wrapper
Ю
regularization_losses
blayer_metrics
clayer_regularization_losses
trainable_variables

dlayers
emetrics
fnon_trainable_variables
	variables
з__call__
ж_default_save_signature
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
-
іserving_default"
signature_map
+:)2conv3d/kernel
:2conv3d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
regularization_losses
glayer_metrics
hlayer_regularization_losses

ilayers
trainable_variables
jmetrics
knon_trainable_variables
	variables
й__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
regularization_losses
llayer_metrics
mlayer_regularization_losses

nlayers
trainable_variables
ometrics
pnon_trainable_variables
	variables
л__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
!regularization_losses
qlayer_metrics
rlayer_regularization_losses

slayers
"trainable_variables
tmetrics
unon_trainable_variables
#	variables
н__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
%regularization_losses
vlayer_metrics
wlayer_regularization_losses

xlayers
&trainable_variables
ymetrics
znon_trainable_variables
'	variables
п__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
А
+regularization_losses
{layer_metrics
|layer_regularization_losses

}layers
,trainable_variables
~metrics
non_trainable_variables
-	variables
с__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
/regularization_losses
layer_metrics
 layer_regularization_losses
layers
0trainable_variables
metrics
non_trainable_variables
1	variables
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
3regularization_losses
layer_metrics
 layer_regularization_losses
layers
4trainable_variables
metrics
non_trainable_variables
5	variables
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
Е
9regularization_losses
layer_metrics
 layer_regularization_losses
layers
:trainable_variables
metrics
non_trainable_variables
;	variables
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
=regularization_losses
layer_metrics
 layer_regularization_losses
layers
>trainable_variables
metrics
non_trainable_variables
?	variables
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Aregularization_losses
layer_metrics
 layer_regularization_losses
layers
Btrainable_variables
metrics
non_trainable_variables
C	variables
ы__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Eregularization_losses
layer_metrics
 layer_regularization_losses
layers
Ftrainable_variables
metrics
non_trainable_variables
G	variables
э__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Iregularization_losses
layer_metrics
 layer_regularization_losses
 layers
Jtrainable_variables
Ёmetrics
Ђnon_trainable_variables
K	variables
я__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Mregularization_losses
Ѓlayer_metrics
 Єlayer_regularization_losses
Ѕlayers
Ntrainable_variables
Іmetrics
Їnon_trainable_variables
O	variables
ё__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 :	Р2x_pred/kernel
:2x_pred/bias
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
Е
Sregularization_losses
Јlayer_metrics
 Љlayer_regularization_losses
Њlayers
Ttrainable_variables
Ћmetrics
Ќnon_trainable_variables
U	variables
ѓ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 :	Р2y_pred/kernel
:2y_pred/bias
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
Е
Yregularization_losses
­layer_metrics
 Ўlayer_regularization_losses
Џlayers
Ztrainable_variables
Аmetrics
Бnon_trainable_variables
[	variables
ѕ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
8
В0
Г1
Д2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

Еtotal

Жcount
З	variables
И	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Э

Йtotal

Кcount
Л	variables
М	keras_api"
_tf_keras_metricx{"class_name": "Mean", "name": "x_pred_loss", "dtype": "float32", "config": {"name": "x_pred_loss", "dtype": "float32"}}
Э

Нtotal

Оcount
П	variables
Р	keras_api"
_tf_keras_metricx{"class_name": "Mean", "name": "y_pred_loss", "dtype": "float32", "config": {"name": "y_pred_loss", "dtype": "float32"}}
:  (2total
:  (2count
0
Е0
Ж1"
trackable_list_wrapper
.
З	variables"
_generic_user_object
:  (2total
:  (2count
0
Й0
К1"
trackable_list_wrapper
.
Л	variables"
_generic_user_object
:  (2total
:  (2count
0
Н0
О1"
trackable_list_wrapper
.
П	variables"
_generic_user_object
0:.2Adam/conv3d/kernel/m
:2Adam/conv3d/bias/m
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
%:#	Р2Adam/x_pred/kernel/m
:2Adam/x_pred/bias/m
%:#	Р2Adam/y_pred/kernel/m
:2Adam/y_pred/bias/m
0:.2Adam/conv3d/kernel/v
:2Adam/conv3d/bias/v
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
%:#	Р2Adam/x_pred/kernel/v
:2Adam/x_pred/bias/v
%:#	Р2Adam/y_pred/kernel/v
:2Adam/y_pred/bias/v
ю2ы
H__inference_functional_1_layer_call_and_return_conditional_losses_389990
H__inference_functional_1_layer_call_and_return_conditional_losses_390291
H__inference_functional_1_layer_call_and_return_conditional_losses_390237
H__inference_functional_1_layer_call_and_return_conditional_losses_389950Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
!__inference__wrapped_model_389624У
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+
input_1џџџџџџџџџ№$@
2џ
-__inference_functional_1_layer_call_fn_390125
-__inference_functional_1_layer_call_fn_390345
-__inference_functional_1_layer_call_fn_390058
-__inference_functional_1_layer_call_fn_390318Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_conv3d_layer_call_and_return_conditional_losses_390356Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_conv3d_layer_call_fn_390365Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
О2Л
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_389630э
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *MЂJ
HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ѓ2 
.__inference_max_pooling3d_layer_call_fn_389636э
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *MЂJ
HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ф2С
C__inference_dropout_layer_call_and_return_conditional_losses_390377
C__inference_dropout_layer_call_and_return_conditional_losses_390382Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
(__inference_dropout_layer_call_fn_390387
(__inference_dropout_layer_call_fn_390392Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
љ2і
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_390398Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
4__inference_tf_op_layer_Reshape_layer_call_fn_390403Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_conv2d_layer_call_and_return_conditional_losses_390414Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_conv2d_layer_call_fn_390423Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Б2Ў
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_389642р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
.__inference_max_pooling2d_layer_call_fn_389648р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ш2Х
E__inference_dropout_1_layer_call_and_return_conditional_losses_390440
E__inference_dropout_1_layer_call_and_return_conditional_losses_390435Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_1_layer_call_fn_390450
*__inference_dropout_1_layer_call_fn_390445Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
D__inference_conv2d_1_layer_call_and_return_conditional_losses_390461Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv2d_1_layer_call_fn_390470Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Г2А
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_389654р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
0__inference_max_pooling2d_1_layer_call_fn_389660р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ш2Х
E__inference_dropout_2_layer_call_and_return_conditional_losses_390487
E__inference_dropout_2_layer_call_and_return_conditional_losses_390482Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
*__inference_dropout_2_layer_call_fn_390492
*__inference_dropout_2_layer_call_fn_390497Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_390503Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_tf_op_layer_Mean_layer_call_fn_390508Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћ2ј
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_390514Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
р2н
6__inference_tf_op_layer_Reshape_1_layer_call_fn_390519Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_390525Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_flatten_layer_call_fn_390530Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_x_pred_layer_call_and_return_conditional_losses_390541Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_x_pred_layer_call_fn_390550Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_y_pred_layer_call_and_return_conditional_losses_390561Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_y_pred_layer_call_fn_390570Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
3B1
$__inference_signature_wrapper_390162input_1Ю
!__inference__wrapped_model_389624Ј
)*78WXQR=Ђ:
3Ђ0
.+
input_1џџџџџџџџџ№$@
Њ "[ЊX
*
x_pred 
x_predџџџџџџџџџ
*
y_pred 
y_predџџџџџџџџџД
D__inference_conv2d_1_layer_call_and_return_conditional_losses_390461l787Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
)__inference_conv2d_1_layer_call_fn_390470_787Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ " џџџџџџџџџ	В
B__inference_conv2d_layer_call_and_return_conditional_losses_390414l)*7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
'__inference_conv2d_layer_call_fn_390423_)*7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ М
B__inference_conv3d_layer_call_and_return_conditional_losses_390356v<Ђ9
2Ђ/
-*
inputsџџџџџџџџџ№$@
Њ "2Ђ/
(%
0џџџџџџџџџ№$@
 
'__inference_conv3d_layer_call_fn_390365i<Ђ9
2Ђ/
-*
inputsџџџџџџџџџ№$@
Њ "%"џџџџџџџџџ№$@Е
E__inference_dropout_1_layer_call_and_return_conditional_losses_390435l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ	
p
Њ "-Ђ*
# 
0џџџџџџџџџ	
 Е
E__inference_dropout_1_layer_call_and_return_conditional_losses_390440l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ	
p 
Њ "-Ђ*
# 
0џџџџџџџџџ	
 
*__inference_dropout_1_layer_call_fn_390445_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ	
p
Њ " џџџџџџџџџ	
*__inference_dropout_1_layer_call_fn_390450_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ	
p 
Њ " џџџџџџџџџ	Е
E__inference_dropout_2_layer_call_and_return_conditional_losses_390482l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
0џџџџџџџџџ
 Е
E__inference_dropout_2_layer_call_and_return_conditional_losses_390487l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
0џџџџџџџџџ
 
*__inference_dropout_2_layer_call_fn_390492_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p
Њ " џџџџџџџџџ
*__inference_dropout_2_layer_call_fn_390497_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ
p 
Њ " џџџџџџџџџЛ
C__inference_dropout_layer_call_and_return_conditional_losses_390377t?Ђ<
5Ђ2
,)
inputsџџџџџџџџџ< 
p
Њ "1Ђ.
'$
0џџџџџџџџџ< 
 Л
C__inference_dropout_layer_call_and_return_conditional_losses_390382t?Ђ<
5Ђ2
,)
inputsџџџџџџџџџ< 
p 
Њ "1Ђ.
'$
0џџџџџџџџџ< 
 
(__inference_dropout_layer_call_fn_390387g?Ђ<
5Ђ2
,)
inputsџџџџџџџџџ< 
p
Њ "$!џџџџџџџџџ< 
(__inference_dropout_layer_call_fn_390392g?Ђ<
5Ђ2
,)
inputsџџџџџџџџџ< 
p 
Њ "$!џџџџџџџџџ< Є
C__inference_flatten_layer_call_and_return_conditional_losses_390525]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ<
Њ "&Ђ#

0џџџџџџџџџР
 |
(__inference_flatten_layer_call_fn_390530P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ<
Њ "џџџџџџџџџРэ
H__inference_functional_1_layer_call_and_return_conditional_losses_389950 
)*78WXQREЂB
;Ђ8
.+
input_1џџџџџџџџџ№$@
p

 
Њ "KЂH
A>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 э
H__inference_functional_1_layer_call_and_return_conditional_losses_389990 
)*78WXQREЂB
;Ђ8
.+
input_1џџџџџџџџџ№$@
p 

 
Њ "KЂH
A>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 ь
H__inference_functional_1_layer_call_and_return_conditional_losses_390237
)*78WXQRDЂA
:Ђ7
-*
inputsџџџџџџџџџ№$@
p

 
Њ "KЂH
A>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 ь
H__inference_functional_1_layer_call_and_return_conditional_losses_390291
)*78WXQRDЂA
:Ђ7
-*
inputsџџџџџџџџџ№$@
p 

 
Њ "KЂH
A>

0/0џџџџџџџџџ

0/1џџџџџџџџџ
 Ф
-__inference_functional_1_layer_call_fn_390058
)*78WXQREЂB
;Ђ8
.+
input_1џџџџџџџџџ№$@
p

 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџФ
-__inference_functional_1_layer_call_fn_390125
)*78WXQREЂB
;Ђ8
.+
input_1џџџџџџџџџ№$@
p 

 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџУ
-__inference_functional_1_layer_call_fn_390318
)*78WXQRDЂA
:Ђ7
-*
inputsџџџџџџџџџ№$@
p

 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџУ
-__inference_functional_1_layer_call_fn_390345
)*78WXQRDЂA
:Ђ7
-*
inputsџџџџџџџџџ№$@
p 

 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџю
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_389654RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ц
0__inference_max_pooling2d_1_layer_call_fn_389660RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџь
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_389642RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
.__inference_max_pooling2d_layer_call_fn_389648RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_389630И_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "UЂR
KH
0Aџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 о
.__inference_max_pooling3d_layer_call_fn_389636Ћ_Ђ\
UЂR
PM
inputsAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HEAџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџм
$__inference_signature_wrapper_390162Г
)*78WXQRHЂE
Ђ 
>Њ;
9
input_1.+
input_1џџџџџџџџџ№$@"[ЊX
*
x_pred 
x_predџџџџџџџџџ
*
y_pred 
y_predџџџџџџџџџА
L__inference_tf_op_layer_Mean_layer_call_and_return_conditional_losses_390503`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
1__inference_tf_op_layer_Mean_layer_call_fn_390508S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџБ
Q__inference_tf_op_layer_Reshape_1_layer_call_and_return_conditional_losses_390514\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ<
 
6__inference_tf_op_layer_Reshape_1_layer_call_fn_390519O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ<П
O__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_390398l;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ< 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
4__inference_tf_op_layer_Reshape_layer_call_fn_390403_;Ђ8
1Ђ.
,)
inputsџџџџџџџџџ< 
Њ " џџџџџџџџџ Ѓ
B__inference_x_pred_layer_call_and_return_conditional_losses_390541]QR0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "%Ђ"

0џџџџџџџџџ
 {
'__inference_x_pred_layer_call_fn_390550PQR0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџЃ
B__inference_y_pred_layer_call_and_return_conditional_losses_390561]WX0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "%Ђ"

0џџџџџџџџџ
 {
'__inference_y_pred_layer_call_fn_390570PWX0Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџ
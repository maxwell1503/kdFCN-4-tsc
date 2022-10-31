window.HELP_IMPROVE_VIDEOJS = false;

var IMAGES_BASE = "./static/images/";
var NUM_IMG_FILTERS = 9;
var NUM_IMG_LAYERS = 2;
var NB_PARAMS_FILTERS = [418, 1338, 2770, 4714, 7170, 10138, 13618, 17610, 67978];
var NB_PARAMS_LAYERS = [2954, 169354];

var interp_images = [];
var interp_images_teacher = [];
var interp_images_wtl = [];
var interp_images_wtl_teacher = [];
var interp_images_layers = [];
var interp_images_layers_teacher = [];
var interp_images_layers_wtl = [];
var interp_images_layers_wtl_teacher = [];

function preloadImages() {
  for (var i = 0; i < NUM_IMG_FILTERS; i++) {
    var path = IMAGES_BASE + '/student_filter_' + String(i+1) + '.png';
    interp_images[i] = new Image();
    interp_images[i].src = path;
    var path_teacher = IMAGES_BASE + '/student_filter_' + String(i+1) + '.png';
    interp_images_teacher[i] = new Image();
    interp_images_teacher[i].src = path_teacher;
    var path_wtl = IMAGES_BASE + '/filters_acc_model' + String(i+1) + '_UCR112.png';
    interp_images_wtl[i] = new Image();
    interp_images_wtl[i].src = path_wtl;
    var path_wtl_teacher = IMAGES_BASE + '/filters_acc_model' + String(i+1) + '_teacher_UCR112.png';
    interp_images_wtl_teacher[i] = new Image();
    interp_images_wtl_teacher[i].src = path_wtl_teacher;
  }
  for (var i = 0; i < NUM_IMG_LAYERS; i++) {
    var path_layers = IMAGES_BASE + '/student_layer_' + String(i+1) + '.png';
    interp_images_layers[i] = new Image();
    interp_images_layers[i].src = path_layers;
    var path_layers_teacher = IMAGES_BASE + '/student_layer_' + String(i+1) + '.png';
    interp_images_layers_teacher[i] = new Image();
    interp_images_layers_teacher[i].src = path_layers_teacher;
    var path_layers_wtl = IMAGES_BASE + '/layers_acc_model' + String(i+1) + '_UCR112.png';
    interp_images_layers_wtl[i] = new Image();
    interp_images_layers_wtl[i].src = path_layers_wtl;
    var path_layers_wtl_teacher = IMAGES_BASE + '/layers_acc_model' + String(i+1) + '_teacher_UCR112.png';
    interp_images_layers_wtl_teacher[i] = new Image();
    interp_images_layers_wtl_teacher[i].src = path_layers_wtl_teacher;
  }
}

function setFilterStudentImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#student_filter_archi-image-wrapper').empty().append(image);

  $('#nb-params-filters').empty().append(NB_PARAMS_FILTERS[i]);

  var image_wtl = interp_images_wtl[i];
  image_wtl.ondragstart = function() { return false; };
  image_wtl.oncontextmenu = function() { return false; };
  $('#student_filter_wtl-image-wrapper').empty().append(image_wtl);
}

function setLayerStudentImage(i) {
  var image_layer = interp_images_layers[i];
  image_layer.ondragstart = function() { return false; };
  image_layer.oncontextmenu = function() { return false; };
  $('#student_layer_archi-image-wrapper').empty().append(image_layer);

  $('#nb-params-layers').empty().append(NB_PARAMS_LAYERS[i]);

  var image_layer_wtl = interp_images_layers_wtl[i];
  image_layer_wtl.ondragstart = function() { return false; };
  image_layer_wtl.oncontextmenu = function() { return false; };
  $('#student_layer_wtl-image-wrapper').empty().append(image_layer_wtl);
}

function setFilterTeacherImage(i) {
  var image_teacher = interp_images_teacher[i];
  image_teacher.ondragstart = function() { return false; };
  image_teacher.oncontextmenu = function() { return false; };
  $('#teacher_filter_archi-image-wrapper').empty().append(image_teacher);

  $('#nb-params-filters-teacher').empty().append(NB_PARAMS_FILTERS[i]);

  var image_wtl_teacher = interp_images_wtl_teacher[i];
  image_wtl_teacher.ondragstart = function() { return false; };
  image_wtl_teacher.oncontextmenu = function() { return false; };
  $('#teacher_filter_wtl-image-wrapper').empty().append(image_wtl_teacher);
}

function setLayerTeacherImage(i) {
  var image_layer_teacher = interp_images_layers_teacher[i];
  image_layer_teacher.ondragstart = function() { return false; };
  image_layer_teacher.oncontextmenu = function() { return false; };
  $('#teacher_layer_archi-image-wrapper').empty().append(image_layer_teacher);

  $('#nb-params-layers-teacher').empty().append(NB_PARAMS_LAYERS[i]);

  var image_layer_wtl_teacher = interp_images_layers_wtl_teacher[i];
  image_layer_wtl_teacher.ondragstart = function() { return false; };
  image_layer_wtl_teacher.oncontextmenu = function() { return false; };
  $('#teacher_layer_wtl-image-wrapper').empty().append(image_layer_wtl_teacher);
}


$(document).ready(function() {
    
    preloadImages();

    $('#student-filter-slider').on('input', function(event) {
      setFilterStudentImage(this.value);
    });
    setFilterStudentImage(0);
    $('#student-filter-slider').prop('max', NUM_IMG_FILTERS - 1);

    $('#student-layer-slider').on('input', function(event) {
      setLayerStudentImage(this.value);
    });
    setLayerStudentImage(0);
    $('#student-layer-slider').prop('max', NUM_IMG_LAYERS - 1);

    $('#teacher-filter-slider').on('input', function(event) {
      setFilterTeacherImage(this.value);
    });
    setFilterTeacherImage(0);
    $('#teacher-filter-slider').prop('max', NUM_IMG_FILTERS - 1);

    $('#teacher-layer-slider').on('input', function(event) {
      setLayerTeacherImage(this.value);
    });
    setLayerTeacherImage(0);
    $('#teacher-layer-slider').prop('max', NUM_IMG_LAYERS - 1);

    bulmaSlider.attach();

})

/* REF CODE: http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/ */
/* Colors from http://flatuicolorpicker.com */
/* Symbols from https://fortawesome.github.io/Font-Awesome/ */
var kit_dict = {
    'tools': [
        ['paint','fa-paint-brush'],
        // ['background','fa-picture-o'],
        ['delete','fa-trash-o'],
        // ['upload','fa-upload'],
    ],
    'paints': [
        ["#000000",'BACKGROUND'],
        ["#800000",'FACE'],
        ["#008000",'HAIR'],
        ["#808000",'EYE'],
        ["#000080",'EYEBROW'],
        ["#800080",'NOSE'],
        ["#008080",'MOUTH'],
        ["#808080",'CLOTH'],
        ["#400000",'BODY'],
    ]
    // 'paints': ["#00B16A","#4ECDC4","#A2DED0","#87D37C","#90C695","#26A65B","#03C9A9","#68C3A3","#65C6BB","#1BBC9B","#1BA39C","#66CC99","#36D7B7","#C8F7C5","#86E2D5","#2ECC71","#16a085","#3FC380","#019875","#03A678","#4DAF7C","#2ABB9B","#1E824C","#049372","#26C281","#446CB3","#E4F1FE","#4183D7","#59ABE3","#81CFE0","#52B3D9","#C5EFF7","#22A7F0","#3498DB","#2C3E50","#19B5FE","#336E7B","#22313F","#6BB9F0","#1E8BC3","#3A539B","#34495E","#67809F","#2574A9","#1F3A93","#89C4F4","#4B77BE","#5C97BF","#EC644B","#D24D57","#F22613","#D91E18","#96281B","#EF4836","#D64541","#C0392B","#CF000F","#E74C3C","#DB0A5B","#F64747","#F1A9A0","#D2527F","#E08283","#F62459","#E26A6A","#DCC6E0","#663399","#674172","#AEA8D3","#913D88","#9A12B3","#BF55EC","#BE90D4","#8E44AD","#9B59B6","#e9d460","#FDE3A7","#F89406","#EB9532","#E87E04","#F4B350","#F2784B","#EB974E","#F5AB35","#D35400","#F39C12","#F9690E","#F9BF3B","#F27935","#E67E22","#ececec","#6C7A89","#D2D7D3","#EEEEEE","#BDC3C7","#ECF0F1","#95A5A6","#DADFE1","#ABB7B7","#F2F1EF","#BFBFBF","#EC644B","#D24D57","#F22613","#D91E18","#96281B","#EF4836","#D64541","#C0392B","#CF000F","#E74C3C","#DB0A5B","#F64747","#F1A9A0","#D2527F","#E08283","#F62459","#E26A6A","#DCC6E0","#663399","#674172","#AEA8D3","#913D88","#9A12B3","#BF55EC","#BE90D4","#8E44AD","#9B59B6","#446CB3","#E4F1FE","#4183D7","#59ABE3","#81CFE0","#52B3D9","#C5EFF7","#22A7F0","#3498DB","#2C3E50","#19B5FE","#336E7B","#22313F","#6BB9F0","#1E8BC3","#3A539B","#34495E","#67809F","#2574A9","#1F3A93","#89C4F4","#4B77BE","#5C97BF","#4ECDC4","#A2DED0","#87D37C","#90C695","#26A65B","#03C9A9","#68C3A3","#65C6BB","#1BBC9B","#1BA39C","#66CC99","#36D7B7","#C8F7C5","#86E2D5","#2ECC71","#16a085","#3FC380","#019875","#03A678","#4DAF7C","#2ABB9B","#00B16A","#1E824C","#049372","#26C281","#e9d460","#FDE3A7","#F89406","#EB9532","#E87E04","#F4B350","#F2784B","#EB974E","#F5AB35","#D35400","#F39C12","#F9690E","#F9BF3B","#F27935","#E67E22","#ececec","#6C7A89","#D2D7D3","#EEEEEE","#BDC3C7","#ECF0F1","#95A5A6","#DADFE1","#ABB7B7","#F2F1EF","#BFBFBF"]
};
var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var clickColor = new Array();
var clickSize = new Array();
var paint;
var toolSelected = 'paint'

// Settings
var strokeColorSetting = "#00b16a";
var strokeSizeSetting = 5;

// Elements
var canvas = document.getElementById("canvas");
var control = document.getElementById("control");
var context = canvas.getContext("2d");
var brushDisplay = document.getElementById('brush');
// var colorDisplay = document.getElementById('color');
var paintList = document.getElementsByClassName("tool-thin");
var toolList = document.getElementsByClassName("tool");
var paintKit = document.getElementById('paints');
var toolKit = document.getElementById('tools');

// Document function
function initializeDocument(){
    // canvas.width = document.body.clientWidth;
    // canvas.height = document.body.clientHeight - control.clientHeight;
    for (var i = 0; i < kit_dict['paints'].length; i++){
        paintKit.innerHTML += "<div class='tool-thin' style='background:" + kit_dict['paints'][i][0] + "!important' " + "color='"+ kit_dict['paints'][i][0] +"'" +">" +
            "<h2>" + kit_dict['paints'][i][1] + "</h2></div>\n";
    }
    for (var i = 0; i < kit_dict['tools'].length; i++){
        toolKit.innerHTML += "<div class='tool' id='" + kit_dict['tools'][i][0] + "'>\n<i class='fa "+ kit_dict['tools'][i][1] +"'></i>\n<h2>" + kit_dict['tools'][i][0] + "</h2></div>\n";
    }
    brushDisplay.setAttribute("style", "border-width: " + brushDisplay.value + "px");
    var paintButton = document.getElementById("paint");
    paintList[0].classList.add("selected");
    paintButton.classList.add("selected");
}

function get_mouse_position(canvas, event) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
};

// Drawing functions
function addClick(x, y, dragging){
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
    clickColor.push(strokeColorSetting);
    clickSize.push(strokeSizeSetting);
}

function resetCanvas(){
    clickX = new Array();
    clickY = new Array();
    clickDrag = new Array();
    clickColor = new Array();
    clickSize = new Array();
    canvas.style.background = "#000000";
}

function clearCanvas(context){
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
}

function redraw(context){
    clearCanvas(context);
    context.lineJoin = "round";
    for (var i = 0; i < clickX.length; i++){
        context.lineWidth = clickSize[i];
        context.beginPath();
        if(clickDrag[i] && i){
            context.moveTo(clickX[i-1], clickY[i-1]);
        }
        else{
            context.moveTo(clickX[i]-1, clickY[i]);
        }
        context.lineTo(clickX[i], clickY[i]);
        context.closePath();
        context.strokeStyle = clickColor[i];
        context.stroke();
    }
}

// main:
initializeDocument();

// Drawing board event functions
function handleStart(event){
    if (toolSelected == 'paint'){
        mouse_position = get_mouse_position(canvas, event);
        var mouseX = mouse_position.x;
        var mouseY = mouse_position.y;
        paint = true;
        addClick(mouse_position.x, mouse_position.y);
        redraw(context);
    }
    // else if (toolSelected == 'background'){
    //     canvas.style.background = strokeColorSetting;
    // }
}

function handleMove(event){
    event.preventDefault();
    if (paint && toolSelected == 'paint'){
        mouse_position = (event.type != "touchmove") ? get_mouse_position(canvas, event) : get_mouse_position(canvas, event.touches[0]);
        addClick(mouse_position.x, mouse_position.y, true)
        redraw(context);
    }
}

function handleEnd(event){
    if (toolSelected == 'paint'){
        paint = false;
    }
}

// Drawing board events
canvas.addEventListener('mousedown', handleStart, false);
canvas.addEventListener('mousemove', handleMove, false);
canvas.addEventListener('mouseup', handleEnd, false);
canvas.addEventListener('mouseleave', handleEnd, false);
canvas.addEventListener("touchstart", handleStart, false);
canvas.addEventListener("touchend", handleEnd, false);
canvas.addEventListener("touchcancel", handleEnd, false);
canvas.addEventListener("touchmove", handleMove, false);

var clearButton = document.getElementById('delete');
clearButton.addEventListener('click', function(event){
    clearCanvas(context);
    resetCanvas(context);
}, false);


var saveButton = document.getElementById('upload');
saveButton.addEventListener('click', function(event){
    var img = canvas.toDataURL("image/png");
    // window.location = img;
    document.getElementById("canvas_base64").value = img.toString();
}, false);

for (var i = 0; i < paintList.length; i++) {
    paintList[i].addEventListener('click', function(event){
        for (var i = 0; i < paintList.length; i++) {
            paintList[i].classList.remove("selected");
        }
        strokeColorSetting = this.getAttribute("color");
        control.setAttribute("style","border-color: " + strokeColorSetting + "!important");
        // colorDisplay.value = strokeColorSetting;
        // colorDisplay.setAttribute("style", "border-color: " + strokeColorSetting + "!important" );
        this.classList.add("selected");
    }, false);
}

brushDisplay.addEventListener('change', function(event){
    strokeSizeSetting = this.value
    this.setAttribute("style", "border-width: " + strokeSizeSetting + "px!important" );
}, false);

// colorDisplay.addEventListener('change', function(event){
//     for (var i = 0; i < paintList.length; i++) {
//         paintList[i].classList.remove("selected");
//     }
//     strokeColorSetting = this.value;
//     control.setAttribute("style","border-color: " + strokeColorSetting + "!important");
//     this.setAttribute("style", "border-color: " + strokeColorSetting + "!important" );
// }, false);

control.addEventListener('click', function(event){
    this.classList.remove('down');
    this.style.cursor = "default";
}, false)

control.addEventListener('mouseleave', function(event){
    this.classList.add('down');
    this.style.cursor = "pointer";
}, false)

for (var i = 0; i < toolList.length; i++) {
    toolList[i].addEventListener('click', function(event){
        for (var i = 0; i < toolList.length; i++) {
            toolList[i].classList.remove("selected");
        }
        toolSelected = this.getAttribute("id");
        this.classList.add("selected");
    }, false);
}
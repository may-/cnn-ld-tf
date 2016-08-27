/* Select Wiki */
$(document).on('click', '.dropdown-menu li', function( event ) {
  var $target = $( event.currentTarget );
  $('.dropdown-menu li.active').toggleClass('active');
  $target.toggleClass('active');
  $target.closest( '.btn-group' )
    .find( '[data-bind="label"]' ).text( $target.text() )
    .end()
    .children( '.dropdown-toggle' ).dropdown( 'toggle' );
  return false;
});

/* Get wiki article*/
$(document).on('click', '#get-wiki', function() {
  var host = $('.dropdown-menu li.active').text().split(" ")[0];
  var base_path = host + "/w/api.php?format=xml&action=query&list=random&rnnamespace=0&rnlimit=1";
  //console.log('host', base_path);

  $.ajax({
    type: "GET",
    url: base_path,
    contentType: "application/json; charset=utf-8",
    async: false,
    dataType: "json",
    success: function (data) {
      alert(data);
    },
    error: function(XMLHttpRequest, textStatus, errorThrown){
      alert('Error : ' + errorThrown);
    }
  });
});

/* Predict */
$(document).on('click', '#send-text', function() {

  var text = $("#get-text").val();
  var host = $("#backend").val();
  //sanitize
  //text = text.replace(/(^[ '\^\$\*#&]+)|([ '\^\$\*#&]+$)/g, '')
  host = host.replace(/(\/)$/, '');
  //console.log(host);

  var pred = '';
  var scores = {};
  $.ajax({
    type: "GET",
    url: host + "/predict?text=" + text,
    contentType: "application/json; charset=utf-8",
    async: false,
    dataType: "json",
    success: function (data) {
      pred = data.prediction;
      scores = data.scores;
    },
    error: function(XMLHttpRequest, textStatus, errorThrown){
      pred = '';
      scores = {};
      alert('Error : ' + errorThrown);
    }
  });
  //console.log(Object.keys(scores).length , scores);

  if (Object.keys(scores).length > 0){
    //alert(pred);
    var dataPoints = [];
    $.each( scores, function( key, value ) {
      //console.log( key + ": " + value );
      dataPoints.push({'y': value, 'label': key });
    });
    dataPoints.sort(
      function(a, b) {
        return b['y'] - a['y']
      }
    );
    $.each( dataPoints, function( i, value ) {
      if (i < 10){
        value['x'] = (i * 10) + 10;
      }
    });
    //console.log(dataPoints.slice(0, 10))
    //console.log(pred);
    //console.log(scores[pred]);
    $('#prediction').text(pred);
    $('#score').text(scores[pred]);


    //draw bar chart
    var chart = new CanvasJS.Chart("chartContainer",
      {
        title:{
          text: "Top 10 Predictions",
          fontFamily: "Helvetica",
          fontSize: 22
        },
        axisX: {
          labelFontFamily: "Helvetica",
          labelFontSize: 14
        },
        axisY: {
          title: "Score",
          titleFontFamily: "Helvetica",
          titleFontSize: 18,
          labelFontFamily: "Helvetica"
        },
        data: [{
          type: "column",

          dataPoints: dataPoints.slice(0, 10)
        }]
      });
    chart.render();
  }

  return false; // abort reload
});

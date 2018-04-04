$(document).ready(function() {
  /* initial rendering */
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
        labelFontFamily: "Helvetica",
        labelFontSize: 14
      },
      data: [{
        type: "column",
        dataPoints: [
          {x: 10, y: 473.721, label: 'Japanese'},
          {x: 20, y: 99.5272, label: 'Greek'},
          {x: 30, y: 76.0632, label: 'Thai'},
          {x: 40, y: 60.9401, label: 'Urdu'},
          {x: 50, y: 52.9983, label: 'Sinhalese'},
          {x: 60, y: 21.5522, label: 'Georgian'},
          {x: 70, y: 20.3253, label: 'Hebrew'},
          {x: 80, y: 11.8332, label: 'Nepali'},
          {x: 90, y: 9.92754, label: 'Mongolian'},
          {x: 100, y: -1.77319, label: 'Telugu'}
        ]
      }]
    });
  chart.render();


  /* Redraw Prediction */
  $('#send-text').on('click', function() {
    var text = $("#get-text").val();
    var host = 'http://localhost:5000';
    //sanitize
    text = text.replace(/(^[ '\^\$\*#&]+)|([ '\^\$\*#&]+$)/g, '')
    host = host.replace(/(\/)$/, '');
    console.log(host + "/predict?text=" + text);

    var pred = '';
    var scores = {};
    $.ajax({
      type: "GET",
      url: host + "/predict?text=" + text,
      contentType: "application/json; charset=utf-8",
      async: false,
      dataType: "json",
      success: function (data) {
        //console.log(data);
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

  $('#set-text_from_file').change(function (event) {
      var filePath = $(this).val();
      $('#filepath').text(filePath);
      jQuery.get(filePath, function(data) {
          console.log(data);
      });
  });
});

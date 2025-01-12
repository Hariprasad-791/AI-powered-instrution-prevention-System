$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    var messages_received = [];
    var ctx = document.getElementById("myChart");
   // Create a pie chart
   var myChart = new Chart(ctx, {
    type: 'pie', 
    data: {
        labels: [], // Labels will be updated dynamically
        datasets: [{
            data: [], // Data values will be updated dynamically
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)',
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)',
            ],
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                display: true,
                position: 'top',
            },
            tooltip: {
                enabled: true,
            }
        }
    }
    });

    //receive details from server
    socket.on('newresult', function(msg) {
        console.log("Received result" + msg.result);
        //maintain a list of ten messages
        if (messages_received.length >= 10){
            messages_received.shift()
        }            
        messages_received.push(msg.result);
        messages_string = '<tr><th>Flow ID</th><th>Src IP</th><th>Src Port</th><th>Dst IP</th><th>Dst Port</th><th>Protocol</th><th>Flow start time</th><th>Flow last seen</th><th>App name</th><th>PID</th><th>Prediction</th><th>Prob</th><th>Risk</th></tr>';

        for (var i = messages_received.length-1 ; i >= 0; i--){
            messages_string = messages_string + '<tr>';
            for (var j = 0; j <messages_received[i].length; j++){
                messages_string = messages_string + '<td>' + messages_received[i][j].toString() + '</td>'; 
            }
            if (messages_received[i][messages_received[i].length - 1] !== "Minimal" && messages_received[i][messages_received[i].length - 3] !== "Benign") {
                let srcIPWithHTML = messages_received[i][1];
                let srcIP = srcIPWithHTML.split(' ')[0]; 
         
                messages_string += `<td>
                    <button onclick="toggleBlock('${srcIP}')">Block</button>
                </td>`;
            } else {
                messages_string += '<td>---</td>';
            }
            messages_string += '</tr>';            
            
        }
        $('#details').html(messages_string);

        for (var i=0; i < msg.ips.length; i++) {
            myChart.data.datasets[0].data[i] =msg.ips[i].count;
            myChart.data.labels[i] =msg.ips[i].SourceIP;
           
           }
           
               myChart.update();

        myChart.update();


    });

});
function toggleBlock(ip) {
    fetch('/toggle-block', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ src_ip: ip }),
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        // Update button text dynamically
        const button = document.querySelector(`button[onclick="toggleBlock('${ip}')"]`);
        button.textContent = data.status === "blocked" ? "Unblock" : "Block";
    })
    .catch(error => {
        console.error('Error:', error);
    });
}





